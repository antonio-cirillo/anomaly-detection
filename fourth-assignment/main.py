import bs4, webbrowser, requests, socket, time
import pandas as pd

sites = []
ip_sites = []

def __getTopSites__():
    url = "https://www.alexa.com/topsites"
    res = requests.get(url)

    alexa = bs4.BeautifulSoup(res.text, 'html.parser')
    
    selector = '.DescriptionCell p a'
    elements = alexa.select(selector)

    for element in elements:
        url = element['href'].replace('/siteinfo/', '')
        try:
            ip = socket.gethostbyname(url)
            ip_sites.append(ip)
            sites.append('http://www.' + url)
        except socket.gaierror:
            print(url + ' not found...')

def __openAllSites__():
    for site in sites:
        webbrowser.open(site)

def __countThirdPartySites__(csv):
    DESTINATION = '172.19.163.142'
    ip_sites.append(DESTINATION)
    COUNTER = 0
    dataFrame = pd.read_csv(csv)
    data = dataFrame.iloc[:, [0, 1, 2, 3, 4]].values
    protocol = [element for element in data[:, 4]]
    source = [element for element in data[:, 2]]
    for s in source:
        if s not in ip_sites:
            COUNTER = COUNTER + 1
            ip_sites.append(s)
    return COUNTER

if __name__ == '__main__':
    __getTopSites__()
    print('IPs and URLs obtained...')
    print('Start capture the traffic...')
    time.sleep(10)
    __openAllSites__()
    print('Capturing...')
    inp = input('Press any key to continue...')
    
    ### TEST 5 MINUTES CAPTURE ###
    print('Test on 5 minutes capture')
    tmp = ip_sites.copy()
    start_time = time.time()
    print('Number of Third Party Sites: ' + str(__countThirdPartySites__('capture-five-minutes.csv')))
    print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')
    
    ### TEST 10 MINUTES CAPTURE ###
    print('Test on 10 minutes capture')
    ip_sites = tmp.copy()
    start_time = time.time()
    print('Number of Third Party Sites: ' + str(__countThirdPartySites__('capture-ten-minutes.csv')))
    print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

    ### TEST 15 MINUTES CAPTURE ###
    print('Test on 15 minutes capture')
    ip_sites = tmp.copy()
    start_time = time.time()
    print('Number of Third Party Sites: ' + str(__countThirdPartySites__('capture-fifteen-minutes.csv')))
    print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

    ### TEST 20 MINUTES CAPTURE ###
    print('Test on 20 minutes capture')
    ip_sites = tmp.copy()
    start_time = time.time()
    print('Number of Third Party Sites: ' + str(__countThirdPartySites__('capture-twenty-minutes.csv')))
    print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

    ### TEST 25 MINUTES CAPTURE ###
    print('Test on 25 minutes capture')
    ip_sites = tmp.copy()
    start_time = time.time()
    print('Number of Third Party Sites: ' + str(__countThirdPartySites__('capture-twentyfive-minutes.csv')))
    print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')