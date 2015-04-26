
# coding: utf-8

# In[1]:

# Importing BeautifulSoup, Pandas, and urllib2 for HTML scraping

from bs4 import BeautifulSoup
from urllib2 import urlopen
import pandas as pd
import numpy as np
import math


# In[2]:

# Function to extract HTML from a Uniform Resource Locator

def make_soup(url):
    html = urlopen(url).read() #Read in HTML and process it into an object
    return BeautifulSoup(html, "lxml") # Parse object using BeautifulSoup


# In[3]:

# Function that crawls through a section of a web page and returns links for each faculty member of the UW-Madison English Dept.

def get_instructor_links(section_url):
    soup = make_soup(section_url) # Create BeautifulSoup HTML object
    ul = soup.find("div", "fac-list") # Set the high-level tag to start the crawl
    
    # Append faculty-specific extensions to the base URL
    instructor_links = [BASE_URL + h3.a["href"] for h3 in ul.findAll("h3")]
    
    return instructor_links


# In[4]:

def get_instructor_info(instructor_link):
        soup = make_soup(instructor_link) # Create BeautifulSoup HTML object
        instructor = soup.find(id="content") # Set the high-level tag in the document to start searches from
        instructor_name = instructor.find("h1").get_text() # Extract instructor names from the 'h1' tag on the web page
        
        
        # Attempt to extract faculty titles from the 'dd' tag on the web page if it exists; if not, return 'None'
        try:
            facultyTitle = instructor.find("dd").get_text()
        except AttributeError:
            return None
        
        # Similarly attempt to extract graduate school info (school name, degree, and year all contained in one 'p' tag)
        try:
            graduate_school = instructor.find("p").get_text()
        except AttributeError:
            return None
        
        return {"full name" : instructor_name,
                "facultytitle": facultyTitle,
                "gradschool": graduate_school}


# In[5]:

BASE_URL = "http://www.english.wisc.edu"

dept_url = "http://www.english.wisc.edu/faculty-by-last-name.htm"

instructors = get_instructor_links(dept_url) # Store links for each faculty member of UW-Madison English Dept.

english_data = [] # a list to store our dictionaries
for instructor in instructors: # Loop through each link
    info = get_instructor_info(instructor) # Extract each instructor's name, title, and grad school info using get_instructor_info function
    english_data.append(info) # Store values in an empty list


# In[6]:

english_data = filter(None, english_data) # Filter out None values in order to convert to dataframe


# In[7]:

english_df = pd.DataFrame(english_data) # Create dataframe

english_df['facultydept'] = 'English' # Add facultydept column


# In[8]:

# Function that crawls through a section of a web page and returns links for each faculty member of the UW-Madison Law School.

def get_instructor_links(dept_url):
    soup = make_soup(dept_url) # Create BeautifulSoup HTML object
    ul = soup.find("table", "directory") # Set the high-level tag to start the crawl
    
    # Append faculty-specific extensions to the base URL
    instructor_links = [td.a["href"] for td in ul.findAll("td", "name")]
    
    return instructor_links


# In[9]:

def get_instructor_info(instructor_link):
        soup = make_soup(instructor_link) # Create BeautifulSoup HTML object
        instructor = soup.find(id="contact") # Set the high-level tag in the document to start searches from
        
        instructor_name = soup.find("h1").get_text() # Extract instructor names from the 'h1' tag on the web page
        
        facultyTitle = soup.find(id="title").get_text() # Extract instructor titles from the tag with id 'title' on the web page
        
        # Extract graduate school info from the second 'p' tag within the tag with id 'contact' on the web page
        for row in instructor.findAll("p")[:2]:
            graduate_school = row.get_text()
        
        return {"full name" : instructor_name,
                "facultytitle": facultyTitle,
                "gradschool": graduate_school}


# In[10]:

dept_url = "http://www.law.wisc.edu/faculty/directory.php?iListing=Faculty&iType=group"

instructors = get_instructor_links(dept_url) # Store links for each faculty member of UW-Madison Law School.

law_data = [] # a list to store our dictionaries
for instructor in instructors: # Loop through each link
    info = get_instructor_info(instructor) # Extract each instructor's name, title, and grad school info using get_instructor_info function
    law_data.append(info) # Store values in an empty list


# In[11]:

law_df = pd.DataFrame(law_data) # Create dataframe

law_df['facultydept'] = 'Law' # Add facultydept column


# In[12]:

# Function that crawls through a section of a web page and returns links for each faculty member of the UW-Madison Chemistry Dept.

def get_instructor_links(section_url):
    soup = make_soup(section_url) # Creating BeautifulSoup HTML object
    ul = soup.find("table", "views-table cols-6") # Setting the high-level tag to start the crawl
    
    # Appending faculty-specific extensions to the base URL
    instructor_links = [BASE_URL + td.a["href"] for td in ul.findAll("td", "views-field views-field-field-first-name")]
    
    return instructor_links


# In[13]:

def get_instructor_info(instructor_link):
        soup = make_soup(instructor_link) # Create BeautifulSoup HTML object
        
        # Set the high-level tag in the document to start the search
        instructor = soup.find("div", "field field-name-field-full-name field-type-text field-label-hidden")
        
        # Extract instructor name from the div under the high-level tag
        instructor_name = instructor.find("div", "field-item even").get_text()
        
        # Set the next high-level tag in the document to start a search
        title = soup.find("div", "field field-name-field-position-name field-type-text field-label-inline clearfix")
        
        # Extract instructor name from the div under the second high-level tag
        facultyTitle = title.find("div", "field-item even").get_text()
        
        # Set the next high-level tag in the document to start a search
        grad = soup.find("div", "field field-name-field-education field-type-text-long field-label-inline clearfix")
        
        # Attempt to extract graduate school info from the div tag on the web page under the high-level tag if it exists; if not, return 'None'
        try:
            graduate_school = grad.find("div", "field-item even").get_text()
        except AttributeError:
            return None
        
        return {"full name" : instructor_name,
                "facultytitle": facultyTitle,
                "gradschool": graduate_school}


# In[14]:

BASE_URL = "http://chem.wisc.edu"

dept_url = "http://chem.wisc.edu/people/faculty"

instructors = get_instructor_links(dept_url) # Store links for each faculty member of UW-Madison Chem Dept.

chem_data = [] # a list to store our dictionaries
for instructor in instructors: # Loop through each link
    info = get_instructor_info(instructor) # Extract each instructor's name, title, and grad school info using get_instructor_info function
    chem_data.append(info) # Store values in an empty list


# In[15]:

chem_data = filter(None, chem_data) # Filter out None values in order to convert to dataframe


# In[16]:

chem_df = pd.DataFrame(chem_data) # Create dataframe

chem_df['facultydept'] = 'Chemistry' # Add facultydept column


# In[49]:

df = english_df.append([law_df, chem_df]) # Merge three departmental dataframes into one


# In[50]:

name_split = df['full name'].str.split() # Split full name column by white space into discrete strings

df['firstname'] = name_split.str[0] # Extract the first element from the string split and store in firstname column

df['lastname'] = name_split.str[-1] # Extract the last element from the string split and store in lastname column


# In[51]:

# Extract any 4-character number value from the gradschool column and store in gradyear
gradyear = df.gradschool.str.extract("(\d{4})")
df['gradyear'] = gradyear

# Extract any instance of 'Ph.D.' and 'J.D.' (and variations) from the gradschool column and store in graddegree
graddegree = df.gradschool.str.extract("(Ph.D.|J.D.|PhD|JD)")
df['graddegree'] = graddegree

# Extract any instance of a string beginning with 'University of' and store in gradschool column
# Note that this now replaces the original gradschool column that contained the raw string data of graduate year, degree, etc.
graduateschool = df.gradschool.str.extract("(University of \w+)")
df['gradschool'] = graduateschool

# Drop the full name column after having extracted distinct first and last names
df = df.drop('full name', 1) 


# In[52]:

df['id'] = range(1, 1 + len(df)) # Create an ID column

df['facultyschool'] = 16 # Create a facultyschool column and add '16' for the University of Wisconsin - Madison

df['facultyyear'] = float('NaN') # Create a facultyyear column and populate it with NaN

df['nametitle'] = float('NaN') # Create a nametitle column and populate it with NaN

df['namesuffix'] = float('NaN') # Create a namesuffix column and populate it with NaN

df['aliases'] = float('NaN') # Create an aliases column and populate it with NaN


# In[53]:

df = df.fillna("") # Convert NaN values to blank spaces


# In[54]:

# Final table output. Note that there is opportunity for column rearranging, depending on need, after downloading the CSV file

df.head()


# In[27]:

# Download and store dataframe as a CSV

df.to_csv("instructor.csv", encoding='utf-8')

