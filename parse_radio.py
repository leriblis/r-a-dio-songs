"""
Scrapes r-a-d.io/last-played for song history and DJ metadata.

Songs are stored in songs_db.json that has the following structure:

{
    # main list containing timestamps and song names
    "songs_dic" : {
        <timestamp, fmt='%Y-%m-%dT%H:%M:%S%z'>: <song>
    },
    # DJ name active during each play event.
    # Same timestamp keys as songs_dic.
    # "Hanyuu-sama" = the automated bot, everything else = a human DJ.
    # May be null/missing for entries scraped before DJ tracking was added,
    # or if the page HTML didn't contain DJ info (very old archives).
    "dj_dic" : {
        <timestamp, fmt='%Y-%m-%dT%H:%M:%S%z'>: <dj_name string or null>
    },
    # some timestamps are broken, we store these songs in a separate list
    # with duplicates removed
    "broken_ts_list" : [
        <{timestamp};{song}>, ...
    ],
    # store the timestamp of the latest song saved
    # only used in update mode
    "latest_ts" : <timestamp, fmt='%Y-%m-%dT%H:%M:%S%z'>,
    # (optional) last URL visited, used for resuming interrupted updates
    "resume_url" : <url string or null>,
    # (optional) resume URL for DJ backfill operation
    "dj_resume_url" : <url string or null>
}

Actions:
    init        - Create a new database from scratch (scrapes all pages)
    update      - Scrape new entries since last run (also captures DJ info)
    resume      - Resume an interrupted init/update
    backfill-dj - Re-scrape all pages to fill in DJ metadata for existing entries.
                  Use --max-pages to limit. Saves progress every 100 pages so you
                  can resume with 'backfill-dj' again. This is idempotent.

"""
import requests, os
from bs4 import BeautifulSoup
import json, time, argparse
from datetime import datetime, timezone
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

RADIO_DATE_FMT = '%Y-%m-%dT%H:%M:%S%z'
PWD = os.path.split(os.path.realpath(__file__))[0]
DBNAME = os.path.join(PWD, 'songs_db.json')
PAGE_PAUSE_TIME = 0.3
LOG = logging.getLogger('parse_radio')

BASE_URL = 'https://r-a-d.io'
FIRST_PAGE_URL = f'{BASE_URL}/last-played?from=4294967295&page=1'

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept":
        "text/html, */*; q=0.01",
    "Accept-Language":
        "en-US,en;q=0.5",
    "Accept-Encoding":
        "gzip, deflate",
}

s = requests.Session()


class ParsingComplete(Exception):
    """Raised when we've reached the endtime boundary."""
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="""Parse last-played songs
        from https://r-a-d.io""")
    parser.add_argument('action', choices=['update', 'init', 'resume', 'backfill-dj'])
    parser.add_argument('--max-pages', type=int, default=0,
                        help='Stop after N pages (0 = no limit)')
    return parser.parse_args()


def get_songs(url, session):
    """
    Get songs from a given page of r-a-d.io/last-played.

    Each entry block on the page contains:
      - A DJ avatar + name (in div.dj-image-name-64 inside div.lp-dj-image)
      - The song title (in the non-narrow column)
      - A unix timestamp (in the <time> element)

    Returns:
        (results, next_url) where results is a list of
        (iso_timestamp, title, dj_name) tuples.
        dj_name is None if the page doesn't contain DJ info (shouldn't happen
        on the current site, but we handle it gracefully).
        next_url is the URL for the next page, or None if no more pages.
    """
    data = session.get(url, headers=HEADERS)
    data.raise_for_status()
    soup = BeautifulSoup(data.text, "html5lib")
    results = []
    for block in soup.select("#page-lastplayed div.block"):
        time_el = block.find('time')
        if not time_el:
            continue
        # Timestamps are now Unix integers
        unix_ts = time_el.get('datetime', '')
        try:
            dt = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
            iso_ts = dt.strftime(RADIO_DATE_FMT)
        except (ValueError, OSError):
            iso_ts = 'nan'

        # Song title is in the non-narrow column with background class
        title_col = block.select_one(
            'div.column.has-background-radio-secondary-1:not(.is-narrow)')
        if not title_col:
            continue
        title = title_col.get_text(strip=True)

        # DJ name is inside the lp-dj-image container.
        # Structure: <div class="lp-dj-image">
        #              <img class="dj-image" src="/api/dj-image/18-xxx.png"/>
        #              <div class="dj-image-name-64">Hanyuu-sama</div>
        #            </div>
        dj_name = None
        dj_el = block.select_one('div.dj-image-name-64')
        if dj_el:
            dj_name = dj_el.get_text(strip=True) or None

        results.append((iso_ts, title, dj_name))

    # Extract next page URL from pagination
    next_url = None
    next_link = soup.select_one('a.pagination-next:not(.is-disabled)')
    if next_link and next_link.get('href'):
        href = next_link['href']
        if not href.startswith('http'):
            href = BASE_URL + href
        next_url = href

    return results, next_url


def save_db(song_db):
    LOG.info("Saving database...")
    LOG.info(f"Database currently contains {len(song_db['songs_dic'])} timestamped songs and")
    LOG.info(f"{len(song_db['broken_ts_list'])} non timestamped songs.")
    dj_dic = song_db.get('dj_dic', {})
    if dj_dic:
        # Count how many entries have actual DJ names vs null
        n_with_dj = sum(1 for v in dj_dic.values() if v is not None)
        LOG.info(f"DJ metadata: {len(dj_dic)} entries ({n_with_dj} with DJ name)")
    with open(DBNAME, 'w') as f:
        json.dump(song_db, f, ensure_ascii=False)


def parse_pages(start_url, endtime, song_db, max_pages=0):
    """
    Scrape songs page by page, following pagination links.

    start_url: str - URL of the first page to scrape
    endtime: timezone-aware datetime - stop when reaching this timestamp
             (None means scrape everything, used for init mode)
    song_db: dict
    max_pages: int - stop after this many pages (0 = no limit)
    """
    songs = song_db['songs_dic']
    dj_dic = song_db.setdefault('dj_dic', {})
    broken_ts_list = song_db['broken_ts_list']
    LOG.info(f"Database contains {len(songs)} timestamped songs and")
    LOG.info(f"{len(broken_ts_list)} non timestamped songs.")
    LOG.info(f"song_db[latest_ts]={song_db['latest_ts']}")
    latest_tmp = song_db['latest_ts']
    current_url = start_url
    page_count = 0
    done = False

    while current_url and not done:
        page_count += 1
        LOG.debug("Getting results from %s", current_url)
        try:
            results, next_url = get_songs(current_url, s)
        except requests.RequestException as e:
            LOG.error(f"Request failed for {current_url}: {e}")
            break
        LOG.debug("A total of %d results obtained", len(results))

        for tmp, song, dj_name in results:
            # Handle broken timestamps
            try:
                dt = datetime.strptime(tmp, RADIO_DATE_FMT)
            except ValueError:
                broken_ts_list.append(f"{tmp};{song}")
                continue

            # Pages go from newest to oldest; stop when we reach known songs
            if endtime is not None and dt <= endtime:
                done = True
                break

            if tmp not in songs:
                songs[tmp] = song
            else:
                if songs[tmp] == song:
                    LOG.debug(f"Duplicate at tmp={tmp}, skipping.")
                else:
                    broken_ts_list.append(f"{tmp};{song}")

            # Always store DJ info if available, even for duplicate song entries.
            # This lets us backfill DJ data for timestamps that already had songs.
            if dj_name is not None:
                dj_dic[tmp] = dj_name

            if tmp > latest_tmp:
                latest_tmp = tmp

        if done:
            break

        if max_pages and page_count >= max_pages:
            LOG.info(f"Reached max pages limit ({max_pages}).")
            break

        current_url = next_url
        time.sleep(PAGE_PAUSE_TIME)

        if page_count % 100 == 0:
            LOG.info(f"Currently on page {page_count}")
            LOG.info(f" latest timestamp is: {latest_tmp}")
            song_db['broken_ts_list'] = list(set(broken_ts_list))
            song_db['resume_url'] = current_url
            save_db(song_db)

    if current_url is None and not done:
        LOG.info("Reached the last page of results.")

    # Update latest_ts and save
    song_db['latest_ts'] = latest_tmp
    song_db['resume_url'] = None
    song_db['broken_ts_list'] = list(set(broken_ts_list))
    save_db(song_db)


def backfill_dj_pages(start_url, song_db, max_pages=0):
    """
    Re-scrape pages to backfill DJ metadata for existing entries.

    Unlike parse_pages(), this does NOT add new songs or update latest_ts.
    It only fills in dj_dic entries. This is idempotent — running it again
    will overwrite DJ names with the same values (harmless).

    Progress is saved every 100 pages via dj_resume_url so you can resume
    an interrupted backfill by running 'backfill-dj' again.

    start_url: str - URL to start from (page 1 or a resume URL)
    song_db: dict
    max_pages: int - stop after N pages (0 = no limit)
    """
    dj_dic = song_db.setdefault('dj_dic', {})
    current_url = start_url
    page_count = 0
    new_dj_entries = 0

    LOG.info(f"DJ backfill starting. Current dj_dic has {len(dj_dic)} entries.")

    # Track consecutive pages with zero DJ info.
    # Once we see many pages in a row without any DJ names, the site
    # probably didn't have DJ metadata that far back — stop early.
    consecutive_empty_pages = 0
    EMPTY_PAGE_THRESHOLD = 50  # 50 pages * 20 entries = 1000 songs with no DJ

    while current_url:
        page_count += 1
        LOG.debug("Getting results from %s", current_url)
        try:
            results, next_url = get_songs(current_url, s)
        except requests.RequestException as e:
            LOG.error(f"Request failed for {current_url}: {e}")
            # Save progress so we can resume from here
            song_db['dj_resume_url'] = current_url
            save_db(song_db)
            break
        LOG.debug("A total of %d results obtained", len(results))

        page_had_dj = False
        for tmp, song, dj_name in results:
            # Skip broken timestamps
            try:
                datetime.strptime(tmp, RADIO_DATE_FMT)
            except ValueError:
                continue

            # Store DJ info if we got a name
            if dj_name is not None:
                page_had_dj = True
                if tmp not in dj_dic:
                    new_dj_entries += 1
                dj_dic[tmp] = dj_name

        # Auto-stop: if many consecutive pages have no DJ data,
        # we've gone past the era where the site tracked DJs.
        if page_had_dj:
            consecutive_empty_pages = 0
        else:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= EMPTY_PAGE_THRESHOLD:
                LOG.info(f"No DJ data found for {EMPTY_PAGE_THRESHOLD} consecutive "
                         f"pages — reached the end of DJ tracking history.")
                song_db['dj_resume_url'] = None
                break

        if max_pages and page_count >= max_pages:
            LOG.info(f"Reached max pages limit ({max_pages}).")
            # Save resume URL so we can continue later
            if next_url:
                song_db['dj_resume_url'] = next_url
            break

        current_url = next_url
        time.sleep(PAGE_PAUSE_TIME)

        if page_count % 100 == 0:
            LOG.info(f"Page {page_count}: {new_dj_entries} new DJ entries "
                     f"({len(dj_dic)} total)")
            song_db['dj_resume_url'] = current_url
            save_db(song_db)

    if current_url is None:
        LOG.info("Reached the last page — DJ backfill complete!")
        song_db['dj_resume_url'] = None

    LOG.info(f"DJ backfill done. Added {new_dj_entries} new entries. "
             f"Total dj_dic: {len(dj_dic)}")
    save_db(song_db)


def main():
    args = parse_arguments()

    if args.action == 'backfill-dj':
        # DJ backfill mode: re-scrape pages to fill in DJ metadata.
        # Does not modify songs_dic or latest_ts — only adds to dj_dic.
        LOG.info("Starting DJ metadata backfill...")
        with open(DBNAME) as f:
            song_db = json.load(f)
        song_db.setdefault('dj_dic', {})
        song_db.setdefault('dj_resume_url', None)

        # Resume from where we left off, or start from page 1
        start_url = song_db.get('dj_resume_url')
        if start_url:
            LOG.info(f"Resuming DJ backfill from: {start_url}")
        else:
            start_url = FIRST_PAGE_URL
            LOG.info("Starting DJ backfill from page 1 (newest).")

        backfill_dj_pages(start_url, song_db, max_pages=args.max_pages)
        return

    if args.action == 'init':
        LOG.info("Initializing new database from start!")
        if os.path.isfile(DBNAME):
            raise ValueError(f"Database file {DBNAME} already exists!")
        # Get the first (newest) timestamp to set as latest_ts
        results, _ = get_songs(FIRST_PAGE_URL, s)
        if not results:
            raise ValueError("Could not get any songs from page 1!")
        newest_ts = results[0][0]
        song_db = {
            "songs_dic": {},
            "dj_dic": {},
            "broken_ts_list": [],
            "latest_ts": newest_ts,
            "resume_url": None,
            "dj_resume_url": None
        }
        start_url = FIRST_PAGE_URL
        endtime = None  # scrape everything
    elif args.action == 'update':
        LOG.info(f"Updating existing database at: {DBNAME}")
        with open(DBNAME) as f:
            song_db = json.load(f)
        song_db.setdefault('resume_url', None)
        song_db.setdefault('dj_dic', {})
        song_db.setdefault('dj_resume_url', None)
        endtime = datetime.strptime(song_db['latest_ts'], RADIO_DATE_FMT)
        start_url = FIRST_PAGE_URL
    elif args.action == 'resume':
        LOG.info("Resuming interrupted update...")
        with open(DBNAME) as f:
            song_db = json.load(f)
        song_db.setdefault('dj_dic', {})
        song_db.setdefault('dj_resume_url', None)
        start_url = song_db.get('resume_url')
        if not start_url:
            LOG.info("No resume URL found, starting from page 1.")
            start_url = FIRST_PAGE_URL
        else:
            LOG.info(f"Resuming from: {start_url}")
        endtime = datetime.strptime(song_db['latest_ts'], RADIO_DATE_FMT)

    original_db_size = len(song_db['songs_dic']) + len(song_db['broken_ts_list'])
    LOG.info(f"Initial db size: {original_db_size}")
    LOG.info(f"Starting at {start_url}")
    if endtime:
        LOG.info(f"Will stop at: {endtime.strftime(RADIO_DATE_FMT)}")
    parse_pages(start_url, endtime, song_db, max_pages=args.max_pages)
    final_db_size = len(song_db['songs_dic']) + len(song_db['broken_ts_list'])
    LOG.info(f"Final db_size: {final_db_size}")
    LOG.info(f"Db increased by: {final_db_size - original_db_size}")


if __name__ == '__main__':
    main()
