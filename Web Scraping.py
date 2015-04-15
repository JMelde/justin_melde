
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
from urllib2 import urlopen


# In[2]:

def make_soup(url):
    html = urlopen(url).read()
    return BeautifulSoup(html, "lxml")


# In[3]:

def get_instructor_links(section_url):
    soup = make_soup(section_url)
    ul = soup.find("div", "fac-list")
    instructor_links = [BASE_URL + h3.a["href"] for h3 in ul.findAll("h3")]
    return instructor_links


# In[4]:

def get_instructor_info(instructor_link):
        soup = make_soup(instructor_link)
        instructor = soup.find(id="content")
        instructor_name = instructor.find("h1").get_text()
        
        try:
            facultyTitle = instructor.find("dd").get_text()
        except AttributeError:
            return None
        
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

instructors = get_instructor_links(dept_url)

data = [] # a list to store our dictionaries
for instructor in instructors:
    info = get_instructor_info(instructor)
    data.append(info)


# In[6]:

data = filter(None, data)


# In[7]:

import pandas as pd
df = pd.DataFrame(data)


# In[8]:

df['id'] = range(70000, 70000 + len(df), 1)

df['facultyschool'] = 16

df['facultydept'] = 'English'

df['facultyyear'] = -1

df['gradyear'] = -1


# In[9]:

df2 = df['full name'].str.split(' ')


# In[10]:

df['firstnames'] = df2.str[0]

df['lastname'] = df2.str[1]


# In[11]:

df = df.drop('full name', 1)


# In[15]:

gradyear = df.gradschool.str.extract("(\d+)")


# In[17]:

df['gradyear'] = gradyear


# In[47]:

graddegree = df.gradschool.str.extract('PhD'|'Ph.D') #or "(Ph.D.)" and "(MA)" or "(MS)" or "(MBA)" or "(MFA)" or "(MM)")


# In[36]:

graddegree.head()


# In[18]:

df.head()


# In[338]:

df.to_csv("instructor.csv", encoding='utf-8')

