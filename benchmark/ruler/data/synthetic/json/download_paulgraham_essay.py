# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import shutil
import glob
import json
import urllib.request
import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm
os.environ["PYTHONHASHSEED"] = str(42)

temp_folder_repo = 'essay_repo'
temp_folder_html = 'essay_html'
os.makedirs(temp_folder_repo, exist_ok=True)
os.makedirs(temp_folder_html, exist_ok=True)

h = html2text.HTML2Text()
h.ignore_images = True
h.ignore_tables = True
h.escape_all = True
h.reference_links = False
h.mark_code = False

# Skip download, use existing local data
print("Using existing local essay data...")

files_repo = glob.glob(os.path.join(temp_folder_repo,'*.txt'))
files_html = glob.glob(os.path.join(temp_folder_html,'*.txt'))
print(f'Download {len(files_repo)} essays from `https://github.com/gkamradt/LLMTest_NeedleInAHaystack/`') 
print(f'Download {len(files_html)} essays from `http://www.paulgraham.com/`') 

text = ""
for file in sorted(files_repo + files_html):    # sort by filename to ensure text is the same
    with open(file, 'r') as f:
        text += f.read()
        
with open('PaulGrahamEssays.json', 'w') as f:
    json.dump({"text": text}, f)


# Keep local directories intact
