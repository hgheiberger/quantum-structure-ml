# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:24:00 2020

@author: linh
"""
import asyncio
import time
import aiohttp
import nest_asyncio
import requests
from bs4 import BeautifulSoup
 
headers = {"Accept-Language": "en-US, en;q=0.5"}
indexes = []
index_sublist = []
mcif_urls = []
 
 
#Adds asyncio support for IDE
nest_asyncio.apply()
 
 
def batch_indexes():
    """
    Scrapes MAGNDATA homepage and appends mcif structure index values
 
    Returns
    -------
    None.
 
    """
    #Pulls datbase homepage through HTML GET request
    url = "http://webbdcrista1.ehu.es/magndata/index.php?show_db=1"
    page = requests.get(url, headers=headers, allow_redirects=True)
 
    #Parses recieved HTML content
    parsed_page = BeautifulSoup(page.text, "lxml")
 
    #Finds and appends mcif index values
    for link in parsed_page.find_all('a'):
        link_text = str(link.get('href'))
        if "index=" in link_text:
            index = link_text.replace("?index=", "")
            indexes.add(str(index))    
 
 
async def identify_structures(structure_index: str):
    """
    Scrapes individual structure database entries and appends mcif download link
 
    Parameters
    ----------
    structure_index : str
        Identification index of indvidual magnetic structure
 
    Returns
    -------
    link : str
        Mcif download link of individual magnetic structure 
 
    """
    #Asynchronous;y requests individual structure database entries
    url = f"http://webbdcrista1.ehu.es/magndata/index_incomm.php?index={structure_index}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            page = await resp.text()
 
            #Parses html response
            parsed_page = BeautifulSoup(page, "lxml")
 
            #Finds and appends mcif download links
            for link in parsed_page.find_all('a'):
                if "mcif" in link.text:
                    mcif_urls.append("http://webbdcrista1.ehu.es/magndata/" + link.get("href"))            
                    link = "http://webbdcrista1.ehu.es/magndata/" + link.get("href")
                    return link
 
 
async def download_data(structure_index: str, link: str):
    """
    Reads individual download links and returns file data
 
    Parameters
    ----------
    structure_index : str
        Identification index of indvidual magnetic structure
    link : str
        Mcif download link of individual magnetic structure 
 
    Returns
    -------
    file_data : bytes
        Mcif file data
 
    """
    #Asynchronously requests individual download links
    url = link
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=1000*60) as resp:
            file_data = await resp.read()
            return file_data
 
 
async def write_to_file(structure_index: str, file_data: bytes):
    """
    Writes file data to a file on the user's local machine
 
    Parameters
    ----------
    structure_index : str
        Identification index of indvidual magnetic structure
    file_data : bytes
        Mcif file data
 
    Returns
    -------
    None.
 
    """
    #Writes file data to a file 
    filename = f"structure_{structure_index}.mcif"
    with open(filename, "wb") as structure_file:
        structure_file.write(file_data)
 
 
async def web_scrape_task(structure_index: str):
    """
    Coordinates coroutines in order to maximize runtime 
 
    Parameters
    ----------
    structure_index : str
        Identification index of indvidual magnetic structure
 
    Returns
    -------
    None.
 
    """
    #Sequentially calls necessary tasks
    link = await identify_structures(structure_index)
    file_data = await download_data(structure_index, link)
    await write_to_file(structure_index, file_data)
 
 
async def main():
    """
    Initializes asyncio response
    Returns
    -------
    None.
 
    """
 
    #Appends indexes to tasklist
    tasks = []
    for index in index_sublist:
        tasks.append(web_scrape_task(index))
    await asyncio.wait(tasks)
 
 
if __name__ == "__main__":
    print("Initializing WebCrawler")
    download_counter = 0
    indexes = set(indexes)
    total_time = time.perf_counter()
 
    #Gathers mcif structure index values
    batch_indexes()
    print(f"Sucessfully located {len(indexes)} mcif structures")
 
 
    #Chunks indexes into sublists for more accurate data processing
    indexes = list(indexes)
    chunks = [indexes[x:x+200] for x in range(0, len(indexes), 200)]    
 
    #Executes coroutines on each sublist
    print("")
    print("Downloading data....")
    for list_index in range(len(chunks)):
 
        index_sublist = chunks[list_index]
        subtask_time = time.perf_counter()
        asyncio.run(main())
        elapsed = time.perf_counter() - subtask_time
        print(f"Downloaded {len(mcif_urls) - download_counter} mcif files in data chunk {list_index + 1} of {len(chunks)}.")
        print(f"Chunk Execution time: {elapsed:0.2f} seconds.")
        download_counter = len(mcif_urls)
 
    print("")
    print(f"Successfully downloaded {len(mcif_urls)} structures with file loss of {len(indexes) - len(mcif_urls)} structures.")
    elapsed = time.perf_counter() - total_time
    print(f"Total Execution time: {elapsed:0.2f} seconds.")