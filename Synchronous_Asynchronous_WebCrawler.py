#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Half-Synchonrous_Half-Asynchronous_WebCrawler
    Finds mcif_strcture url sequentially and downloads mcif_structure data in parallel

@author: hgheiberger
"""

import requests
from bs4 import BeautifulSoup
import urllib.request
import asyncio
import aiohttp
import async_timeout
import time
import nest_asyncio
 
headers = {"Accept-Language": "en-US, en;q=0.5"}
indexes = {"2.1.1", "3.4", "0.4", "0.5", "0.1"} 
current_indexes = []
mcif_urls = ["http://webbdcrista1.ehu.es/magndata/tmp/0.409_TmNi.mcif"]

#Adds asyncio support for IDE
nest_asyncio.apply()

async def get_url(url, session):
    
    """
    Scrapes individual structure database entries and appends mcif download link
 
    Parameters
    ----------
    url : str
        mcif link of individual magnetic structure
    session
 
    Returns
    -------
    "Finished writing" + file_name : str
        Message that the file has been downloaded.
 
    """
    
    file_name = url.split('/')[-1]
    async with async_timeout.timeout(500):
        async with session.get(url) as response:
            with open(file_name, 'wb') as fd:
                async for data in response.content.iter_chunked(1024):
                    fd.write(data)
    return 'Finished writing ' + file_name


async def main(urls):
    """
 
    Parameters
    ----------
    urls : list
        list of mcif urls
 
    Returns
    -------
    Makes a queue of all the urls.
 
    """
    
    async with aiohttp.ClientSession() as session:
        tasks = [get_url(url, session) for url in urls]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
 
    #Pulls datbase homepage through HTML GET request
    url = "http://webbdcrista1.ehu.es/magndata/index.php?show_db=1"
    page = requests.get(url, headers=headers, timeout=10.00, allow_redirects=True)
 
    #Parses recieved HTML content
    parsed_page = BeautifulSoup(page.content,'lxml') 
 
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
        parsed_page = BeautifulSoup(page.content,'lxml') 
 
        #Finds and appends mcif download links
        for link in parsed_page.find_all('a'):
            if "mcif" in link.text:
                mcif_urls.append("http://webbdcrista1.ehu.es/magndata/" + link.get("href"))

    #starts the coroutine
    loop = asyncio.get_event_loop()
    #reulsts holds the messages that the files have been downloaded
    #loop is executed using the run_until_complete method of asyncio
    results = loop.run_until_complete(main(mcif_urls))
    #joins together and prints out all the messages that the files have been downloaded
    print('\n'.join(results))

