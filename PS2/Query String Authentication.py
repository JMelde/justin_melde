
# coding: utf-8

# In[1]:

# Import S3 Connections

from boto.s3.connection import S3Connection


# In[3]:

# Create access key variables

# Note that I've obscured the keys for privacy

access_key = "enter access key here"

secret_key = "enter secrete key here"


# In[4]:

# Connect to S3

conn = S3Connection(access_key, secret_key, is_secure = False)


# In[5]:

# Access my bucket

bucket = conn.get_bucket('jmelde-bucket')


# In[11]:

# Create a key for the 'summary_trigrams' file

ngram_key = bucket.get_key('ps2_output/summary_trigrams.csv')


# In[12]:

# Change the access privileges to private

ngram_key.set_canned_acl('private')


# In[13]:

# Generate the URL

ngram_url = ngram_key.generate_url(864000, query_auth = True, force_http = True)


# In[14]:

# Print the URL

print ngram_url

