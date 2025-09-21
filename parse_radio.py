"""

Songs are stored in songs_db.json that has the following structure:

{
    # main list containing timestamps and song names
    "songs_dic" : {
        <timestamp, fmt='%Y-%m-%dT%H:%M:%S%z'>: <song>
    },
    # some timestamps are broken, we store these songs in a separate list
    # with duplicates removed
    "broken_ts_list" : [
        <{timestamp};{song}>, ...
    ],
    # store the timestamp of the latest song saved
    # only used in update mode
    "latest_ts" : <timestamp, fmt='%Y-%m-%dT%H:%M:%S%z'>
}

"""
import requests, os
from bs4 import BeautifulSoup
import json, time, argparse
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

RADIO_DATE_FMT = '%Y-%m-%dT%H:%M:%S%z'
PWD = os.path.split(os.path.realpath(__file__))[0]
DBNAME = os.path.join(PWD, 'songs_db.json')
PAGE_PAUSE_TIME = 0.3
LOG = logging.getLogger('parse_radio')

HEADERS = {
    "Host":
        "r-a-d.io",
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0",
    "Accept":
        "text/html, */*; q=0.01",
    "Accept-Language":
        "en-US,en;q=0.5",
    "Accept-Encoding":
        "gzip, deflate",
    "Referer":
        "http://r-a-d.io/last-played?page=1",
    "X-Requested-With":
        "XMLHttpRequest"
}

s = requests.Session()
# init cache
s.get("http://r-a-d.io/last-played")


def parse_arguments():
    parser = argparse.ArgumentParser(description="""Parse last-played songs
        from https://r-a-d.io""")
    parser.add_argument('action', choices=['update', 'init', 'resume'])
    parser.add_argument('--page', type=int, help='Only used if action = resume')
    return parser.parse_args()


def parse_page_one():
    p1 = s.get("http://r-a-d.io/last-played?page=1", headers=HEADERS)
    soup = BeautifulSoup(p1.text, "html5lib")
    ts1 = soup.find('li').find('span')['title'] # 1st timestamp
    last_page = soup.select('ul[class="pagination"]')[0].find_all("li")[-2].text
    return ts1, int(last_page)


def get_songs(page, session):
    """
    Get songs from a given page
    page: str
        url of page on radio
    session: requests object
    """
    data = session.get(page, headers=HEADERS)
    soup = BeautifulSoup(data.text, "html5lib")
    results = []
    for l in soup.select("li[class='list-group-item']"):
        title = l.find('span').text
        song_time = l.find('time').get('datetime','nan')
        results.append((song_time, title))
    return results


def save_db(song_db):
    LOG.info("Saving database...")
    LOG.info(f"Database currently contains {len(song_db['songs_dic'])} timestamped songs and")
    LOG.info(f"{len(song_db['broken_ts_list'])} non timestamped songs.")
    with open(DBNAME, 'w') as f:
        json.dump(song_db, f, ensure_ascii=False)


def parse_pages(start_page, endtime, song_db, forward_direction=True):
    """
    Save songs to db

    start_page: int
    endtime: timezone aware datetime object
    song_db: dic
    """
    page_limits = (0, 75000)
    curpage = start_page
    songs = song_db['songs_dic']
    broken_ts_list = song_db['broken_ts_list']
    LOG.info(f"Database contains {len(songs)} timestamped songs and")
    LOG.info(f"{len(broken_ts_list)} non timestamped songs.")
    LOG.info(f"song_db[latest_ts]={song_db['latest_ts']}")
    latest_tmp = song_db['latest_ts']
    while page_limits[0] < curpage < page_limits[1]:
        url = f"http://r-a-d.io/last-played?page={curpage}"
        LOG.debug("Getting results from page %d", curpage)
        results = get_songs(url, s)
        LOG.debug("A total of %d results obtained", len(results))
        # parse songs
        try:
            for tmp, song in results:
                # check if we are over limit
                dt = datetime.strptime(tmp, RADIO_DATE_FMT)
                if forward_direction and dt <= endtime:
                    # forward = we move from newer to older songs
                    raise ValueError("Parsing finished!")
                if not forward_direction and dt > endtime:
                    # backward = we move from older to newer
                    raise ValueError("Parsing finished!")
                if tmp not in songs:
                    songs[tmp] = song
                else:
                    if songs[tmp] == song:
                        LOG.debug(f"Duplicate at tmp={tmp}, skipping.")
                        continue
                    else:
                        broken_ts_list.append(f"{tmp};{song}")
                if tmp > latest_tmp:
                    latest_tmp = tmp
        except ValueError:
            # we are done with the parsing
            break
        time.sleep(PAGE_PAUSE_TIME)
        if forward_direction:
            curpage += 1
        else:
            curpage -= 1
        if curpage % 100 == 0:
            # delete duplicates from broken_ts_list
            LOG.info(f"Currently on page {curpage}")
            LOG.info(f" latest timestamp is: {latest_tmp}")
            song_db['broken_ts_list'] = list(set(broken_ts_list))
            save_db(song_db)
    else:
        LOG.warn(f"""Page {curpage} outside of limits! There is might be a problem with
limits or with the code!""")
    # save last time
    if forward_direction:
        # update timestamp
        song_db['latest_ts'] = latest_tmp
    save_db(song_db)


def main():
    args = parse_arguments()
    if args.action == 'init':
        LOG.info("Initializing new database from start!")
        if os.path.isfile(DBNAME):
            raise ValueError(f"Database file {DBNAME} already exists!")
        endtime, start_page = parse_page_one()
        endtime = datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%S%z')
        song_db = {
            "songs_dic" : {},
            "broken_ts_list": [],
            "latest_ts": endtime.strftime(RADIO_DATE_FMT)
        }
        forward_direction = False
    elif args.action == 'update':
        LOG.info(f"Updating existing database at: {DBNAME}")
        with open(DBNAME) as f:
            song_db = json.load(f)
        forward_direction = True
        endtime = datetime.strptime(song_db['latest_ts'], '%Y-%m-%dT%H:%M:%S%z')
        start_page = 1
    elif args.action == 'resume':
        LOG.info(f"Resuming update after interruption from page: {args.page}")
        with open(DBNAME) as f:
            song_db = json.load(f)
        start_page = args.page - 1
        forward_direction = True
        endtime = datetime.strptime(song_db['latest_ts'], '%Y-%m-%dT%H:%M:%S%z')

    original_db_size = len(song_db['songs_dic']) + len(song_db['broken_ts_list'])
    LOG.info(f"Initial db size: {original_db_size}")
    LOG.info(f"Starting at page={start_page} and ending at {endtime.strftime(RADIO_DATE_FMT)}")
    parse_pages(start_page, endtime, song_db, forward_direction=forward_direction)
    final_db_size = len(song_db['songs_dic']) + len(song_db['broken_ts_list'])
    LOG.info(f"Final db_size: {final_db_size}")
    LOG.info(f"Db increased by: {final_db_size-original_db_size}")


if __name__ == '__main__':
    main()

