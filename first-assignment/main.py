import pyshark
import socket
import csv

URLS = ['google.com', 'youtube.com', 'tmall.com', 'qq.com', 'baidu.com', 'sohu.com',
    'facebook.com', 'taobao.com', '360.cn', 'jd.com', 'amazon.com', 'yahoo.com', 
    'wikipedia.org', 'weibo.com', 'sina.com.cn', 'xinhuanet.com', 'zoom.us', 'live.com', 
    'netflix.com', 'microsoft.com', 'reddit.com', 'instagram.com', 'office.com', 
    'google.com.hk', 'alipay.com', 'bing.com', 'csdn.net', 'myshopify.com', 'vk.com', 
    'bongacams.com', 'yahoo.co.jp', 'twitter.com', 'naver.com', 'okezone.com',
    'twitch.tv','amazon.in', 'ebay.com', 'aparat.com', 'force.com', 'yy.com', 'tianya.cn',
    'adobe.com', 'huanqiu.com', 'chaturbate.com', 'aliexpress.com','linkedin.com', 'amazon.co.jp']

PATH = 'D:\\Università\\Reti Geografiche\\Laboratorio\\cattura.pcapng'

RESULT = []
INDEX = 0
FLAG = False

with open('result.csv', 'w', encoding = 'UTF8') as F:
    WRITER = csv.writer(F)
    WRITER.writerow(['url', 'version'])
    for URL in URLS:
        IP_LIST =  list({ADDR[-1][0] for ADDR in socket.getaddrinfo('www.' + URL, 0, 0, 0, 0)})
        for IP in IP_LIST:
            print('IP: ' + IP + ', URL: ' + URL)
            CAPTURE = pyshark.FileCapture(PATH, display_filter = 'ip.dst == ' + IP)
            try:
                PACKAGE = CAPTURE.next()[0]
                CAPTURE.close()
                CAPTURE = pyshark.FileCapture(PATH, display_filter = 'ip.dst == ' + IP + ' && quic')
                try:
                    PACKAGE = CAPTURE.next()[0]
                    CAPTURE.close()
                    if not FLAG:
                        RESULT.append([URL, 3])
                        break
                    else:
                        RESULT[INDEX] = [URL, 3]
                except:
                    CAPTURE.close()
                    CAPTURE = pyshark.FileCapture(PATH, display_filter = 'ip.dst == ' + IP + ' && http2')
                    try:
                        PACKAGE = CAPTURE.next()[0]
                        CAPTURE.close()
                        if(FLAG):
                            if(RESULT[INDEX][1] < 2):
                                RESULT = [URL, 2]
                        else:
                            RESULT.append([URL, 2])
                            FLAG = True
                    except:
                        CAPTURE.close()
                        if not FLAG:
                            RESULT.append([URL, 1.1])
                            FLAG = True
            except:
                CAPTURE.close()

        try:
            print(RESULT[INDEX])
        except:
            RESULT.append([URL, '-'])
        WRITER.writerow([RESULT[INDEX][0], 'HTTP/' + str(RESULT[INDEX][1])])
        INDEX = INDEX + 1
        FLAG = False