import requests
from bs4 import BeautifulSoup
import os, re
import csv


data_directory = "./Input/gm/"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)


def createEntry(entry):
    
    volume = re.search('(?<=-VOL-)(.*?)(?=-PAGINAS-)', entry, re.DOTALL)
    if volume is not None:
        volume = volume.group(0).strip('\n')
    page = re.search('(?<=-PAGINAS-)(.*?)(?=-AUTOR-)', entry, re.DOTALL)
    if page is not None:
        page = page.group(0).strip('\n')
    author = re.search('(?<=-AUTOR-)(.*?)(?=-TITULO-)', entry, re.DOTALL)
    if author is not None:
        author = author.group(0).strip('\n')
    title = re.search('(?<=-TITULO-)(.*?)(?=-TERMS-)', entry, re.DOTALL)
    if title is not None:
        title = title.group(0).strip('\n')
    key_terms = re.search('(?<=-TERMS-)(.*?)(?=-DESC)', entry, re.DOTALL)
    if key_terms is not None:
        key_terms = key_terms.group(0).strip('\n')
    date = re.search('(?<=-FECHA-)(.*?)(?=-END-)', entry, re.DOTALL)
    if date is not None:
        date = date.group(0).strip('\n')
    
    return [volume, page, author, title, key_terms, date]


for y in list(range(1965, 1993)):

    data_list = []

    url = 'http://lanic.utexas.edu/project/granma/' + str(y) + '.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.content.decode('utf-8','ignore'), 'lxml')
    page_text = soup.find('pre')
    pure_text = page_text.string
    entry_list = pure_text.split("-FUENTE-\nGranma")
    entry_list.pop(0)
    
    for entry in entry_list:
        entry = createEntry(entry)
        data_list.append(entry)
        fname = data_directory + 'Granma_data_' + str(y) + '.csv'
    with open (fname, 'w', encoding='utf-8') as csvfile:
        r_writer=csv.writer(csvfile, delimiter=',', lineterminator='\n')
        r_writer.writerow(["volume", "page", "author", "title", "key_terms", "date"])
        for x in data_list:
            r_writer.writerow(x)