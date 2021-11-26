"""A Simple Script for Extracting Data from a Webpage 
This script allows the user to extract data from a webapge and then export the data to a csv file with column(s).
"""
# libraries
import urllib.request
from bs4 import BeautifulSoup
import csv
# Put your URL here
url = 'https://www.alexa.com/topsites'
# Fetching the html
request = urllib.request.Request(url)
content = urllib.request.urlopen(request)
# Parsing the html 
parse = BeautifulSoup(content, 'html.parser')
text1 = parse.find_all('a')

#for line in text1:
    #print(line.__str__)

# Writing extracted data in a csv file
with open('index.csv', 'w') as csv_file:
  writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
  writer.writerow(['URL'])
  for col1 in zip(text1):
    writer.writerow([col1])