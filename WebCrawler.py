# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 05:31:28 2020

@author: hgheiberger
"""

import requests
from bs4 import BeautifulSoup

headers = {"Accept-Language": "en-US, en;q=0.5"}
indexes = {"2.1.1", "3.4"} 
mcif_urls = ["http://webbdcrista1.ehu.es/magndata/tmp/0.409_TmNi.mcif"]


#Pulls datbase homepage through HTML GET request
url = "http://webbdcrista1.ehu.es/magndata/index.php?show_db=1"
page = requests.get(url, headers=headers, timeout=10.00, allow_redirects=True)

#Parses recieved HTML content
parsed_page = BeautifulSoup(page.text, "lxml")
    
#Finds and appends mcif index values
for link in parsed_page.find_all('a'):
    link_text = str(link.get('href'))
    if "index=" in link_text:
        index = link_text.replace("?index=", "")
        indexes.add(str(index))


#Scrapes individual database entries
for index in indexes:
    #Builds structure urls
    try:
        #Incommensurate structure support   
        if index[1] == "." and index[2] == "1" and index[3] == ".":
            #Pulls webpage through HTML GET request
            url = "http://webbdcrista1.ehu.es/magndata/index_incomm.php?index=" + str(index)
            page = requests.get(url, headers=headers, timeout=10.00) 
        #Commensurate structure support
        else:
            url = "http://webbdcrista1.ehu.es/magndata/index.php?index=" + str(index)
    except:
        url = "http://webbdcrista1.ehu.es/magndata/index.php?index=" + str(index)
   
    #Pulls structure page through HTML GET request
    page = requests.get(url, headers=headers, timeout=10.00)

    #Parses recieved HTML content
    parsed_page = BeautifulSoup(page.text, "lxml")
    
    #Finds and appends mcif download links
    for link in parsed_page.find_all('a'):
        if "mcif" in link.text:
            mcif_urls.append("http://webbdcrista1.ehu.es/magndata/" + link.get("href"))
            
            
            
