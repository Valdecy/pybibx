############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# pyBibX - A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools

# Citation:
# PEREIRA, V.; BASILIO, M.P.; SANTOS, C.H.T. (2025). PyBibX: A Python Library for Bibliometric and
# Scientometric Analysis Powered with Artificial Intelligence Tools. Data Technologies and Applications.
# Vol. 59, Iss. 2, pp. 302-337. doi: https://doi.org/10.1108/DTA-08-2023-0461

############################################################################

# Required Libraries
import chardet
import google.generativeai as genai
import networkx as nx             
import numpy as np   
import openai       
import os       
import pandas as pd     
import PIL 

try:
    pil_version = tuple(map(int, PIL.__version__.split('.')[:3]))
    if pil_version >= (10, 0, 0):
        import PIL.ImageDraw
        if not hasattr(PIL.ImageDraw.ImageDraw, 'textsize'):
            def textsize(self, text, font = None, *args, **kwargs):
                bbox   = self.textbbox((0, 0), text, font = font, *args, **kwargs)
                width  = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return (width, height)
            PIL.ImageDraw.ImageDraw.textsize = textsize
except Exception:
    pass  
      
import plotly.graph_objects as go
import plotly.subplots as ps      
import plotly.io as pio         
import re                                         
import unicodedata                
import textwrap

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from . import stws

from bertopic import BERTopic                               
from collections import Counter, defaultdict
from difflib import SequenceMatcher
#from keybert import KeyBERT 
from gensim.models import FastText
from itertools import combinations
from matplotlib import pyplot as plt                       
plt.style.use('bmh') 
from numba import njit
from numba.typed import List
#from rapidfuzz import fuzz
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer                    
from sklearn.cluster import KMeans, HDBSCAN                          
from sklearn.decomposition import TruncatedSVD as tsvd      
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity  
from summarizer import Summarizer
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from umap import UMAP  
from wordcloud import WordCloud                          

############################################################################

@njit
def build_edges(idx_list):
    total_pairs = 0
    for items in idx_list:
        n = len(items)
        if (n > 1):
            total_pairs = total_pairs + n * (n - 1)
    rows = np.empty(total_pairs, dtype = np.int32)
    cols = np.empty(total_pairs, dtype = np.int32)
    pos  = 0
    for items in idx_list:
        n = len(items)
        for i in range(0, n):
            for j in range(i+1, n):
                a1        = items[i]
                a2        = items[j]
                rows[pos] = a1
                cols[pos] = a2
                pos       = pos + 1
                rows[pos] = a2
                cols[pos] = a1
                pos       = pos + 1
    return rows, cols

@njit
def build_edges_ref(ref_idx_list):
    count = 0
    for row in range(0, len(ref_idx_list)):
        count = count + len(ref_idx_list[row])
    row_indices = np.empty(count, dtype=np.int32)
    col_indices = np.empty(count, dtype=np.int32)
    pos         = 0
    for row in range(0, len(ref_idx_list)):
        refs = ref_idx_list[row]
        for col in refs:
            row_indices[pos] = row
            col_indices[pos] = col
            pos              = pos + 1
    return row_indices, col_indices
            
# pbx Class
class pbx_probe():
    def __init__(self, file_bib, db = 'scopus', del_duplicated = True):
        db                     = db.lower()
        self.database          = db
        self.institution_names =  [ 
                                   'acad', 'academy', 'akad', 'aachen', 'assoc', 'cambridge',
                                   'cefet', 'center', 'centre', 'ctr',  'chuo kikuu', 'cient', 'cirad',
                                   'coll', 'college', 'colegio', 'companhia', 'communities', 'conservatory', 
                                   'council', 'dept', 'egyetemi', 'escola', 'education', 'escuela', 
                                   'embrapa', 'espm', 'epamig','epagri',  'eyunivesithi', 'fac', 
                                   'faculdade', 'facultad', 'fakultet', 'fakultät', 'fal', 'fdn', 
                                   'fundacion', 'foundation', 'fundacao', 'gradevinski', 'grp', 'higher', 
                                   'hsch', 'hochschule', 'hosp', 'hgsk', 'hogeschool',  'háskóli',  
                                   'högskola', 'ibmec', 'ird', 'inivèsite', 'ist', 'istituto', 'imd', 
                                   'institutional', 'int', 'inst',  'institut', 'institute', 
                                   'institute of technology',  'inyuvesi', 'iskola', 'iunivesite', 
                                   'inrae','jaamacad', "jami'a",  'kolej', 'koulu', 'kulanui', 'lab.', 
                                   'lab', 'labs', 'laborat', 'learning', 'mahadum', 'med', 'medicine', 
                                   'medical', 'museum','observatory', 'oilthigh', 'okulu', 'ollscoile', 
                                   'oniversite', 'politecnico', 'polytechnic', 'prifysgol', 'project', 
                                   'rech', 'recherche', 'research', 'sch', 'school', 'schule', 'scuola', 
                                   'seminary', 'skola', 'supérieur', 'sveučilište', 'szkoła', 'tech', 
                                   'technical', 'technische', 'technique', 'technological', 'uff', 'ufrrj', 
                                   'ufruralrj', 'ufmg', 'ufpb', 'ufpe','ufal', 'uned', 'unep', 'unesp', 
                                   'unibersidad', 'unibertsitatea',  'unicenp','ucpel', 'usp', 'ufac', 
                                   'udesc', 'uerj','univ', 'universidad', 'universidade', 'universitas', 
                                   'universitat', 'universitate', 'universitato', 'universite', 
                                   'universiteit', 'universitet', 'universitetas', 'universiti', 
                                   'university', 'università', 'universität', 'université', 'universitāte', 
                                   'univerza', 'univerzita','univerzitet', 'univesithi', 'uniwersytet', 
                                   'vniuersitatis', 'whare wananga', 'yliopisto','yunifasiti', 'yunivesite', 
                                   'yunivhesiti', 'zanko', 'école', 'ülikool', 'üniversite','πανεπιστήμιο', 
                                   'σχολείο', 'универзитет', 'университет', 'універсітэт', 'школа'
                                  ]
        self.inst_priority   =    {
                                   'acad': 30, 'academy': 40, 'akad': 30, 'aachen': 50, 'assoc': 20, 
                                   'cambridge': 90, 'ctr': 20, 'cefet': 50, 'center': 60, 'centre': 60, 
                                   'chuo kikuu': 50, 'cient': 20, 'cirad': 50, 'coll': 40, 'college': 60, 
                                   'colegio': 40, 'companhia': 10, 'communities': 10, 'conservatory': 40, 
                                   'council': 30, 'dept': 20, 'egyetemi': 50, 'escola': 50, 'education': 40, 
                                   'escuela': 50, 'embrapa': 70, 'espm': 60, 'epamig': 60, 'epagri': 60, 
                                   'eyunivesithi': 50, 'fac': 40, 'faculdade': 50, 'facultad': 50, 
                                   'fakultet': 50, 'fakultät': 50, 'fal': 20, 'fdn': 20, 'fundacion': 40, 
                                   'foundation': 40, 'fundacao': 40, 'gradevinski': 30, 'grp': 20, 
                                   'higher': 40, 'hsch': 50, 'hochschule': 50, 'hosp': 30, 'hgsk': 50, 
                                   'hogeschool': 50, 'háskóli': 50, 'högskola': 50, 'ibmec': 60, 'ird': 50, 
                                   'inivèsite': 50, 'ist': 50, 'istituto': 50, 'imd': 60, 
                                   'institutional': 30, 'int': 20, 'inst': 40, 'institut': 70, 
                                   'institute': 90, 'institute of technology': 100, 'inyuvesi': 50, 
                                   'iskola': 50, 'iunivesite': 50, 'inrae': 70, 'jaamacad': 50, 
                                   "jami'a": 50, 'kolej': 50, 'koulu': 50, 'kulanui': 50, 'lab.': 30, 
                                   'lab': 30, 'labs': 30, 'laborat': 40, 'learning': 20, 'mahadum': 50, 
                                   'med': 30, 'medicine': 70, 'medical': 70, 'museum': 40, 
                                   'observatory': 50, 'oilthigh': 50, 'okulu': 50, 'ollscoile': 50, 
                                   'oniversite': 50, 'politecnico': 70, 'polytechnic': 70, 'prifysgol': 70, 
                                   'project': 30, 'rech': 40, 'recherche': 40, 'research': 90, 'sch': 40, 
                                   'school': 60, 'schule': 50, 'scuola': 50, 'seminary': 40, 'skola': 50, 
                                   'supérieur': 50, 'sveučilište': 50, 'szkoła': 50, 'tech': 70, 
                                   'technical': 70, 'technische': 70, 'technique': 70, 'technological': 70, 
                                   'uff': 50, 'ufrrj': 50, 'ufruralrj': 50, 'ufmg': 50, 'ufpb': 50, 
                                   'ufpe': 50, 'ufal': 50, 'uned': 50, 'unep': 50, 'unesp': 50, 
                                   'unibersidad': 50, 'unibertsitatea': 50, 'unicenp': 50, 'ucpel': 50, 
                                   'usp': 70, 'ufac': 50, 'udesc': 50, 'uerj': 50, 'univ': 80, 
                                   'universidad': 90, 'universidade': 90, 'universitas': 90, 
                                   'universitat': 90, 'universitate': 90, 'universitato': 90, 
                                   'universite': 90, 'universiteit': 90, 'universitet': 90, 
                                   'universitetas': 90, 'universiti': 90, 'university': 100, 
                                   'università': 90, 'universität': 90, 'université': 90, 
                                   'universitāte': 90, 'univerza': 90, 'univerzita': 90, 
                                   'univerzitet': 90, 'univesithi': 90, 'uniwersytet': 90, 
                                   'vniuersitatis': 90, 'whare wananga': 50, 'yliopisto': 50, 
                                   'yunifasiti': 50, 'yunivesite': 50, 'yunivhesiti': 50, 
                                   'zanko': 50, 'école': 50, 'ülikool': 50,'üniversite': 90, 
                                   'πανεπιστήμιο': 90, 'σχολείο': 50, 'универзитет': 90, 
                                   'университет': 90, 'універсітэт': 90,'школа': 50
                                  }

        self.language_names  =    { 
                                   'afr': 'Afrikaans', 'alb': 'Albanian','amh': 'Amharic', 'ara': 'Arabic', 
                                   'arm': 'Armenian', 'aze': 'Azerbaijani', 'bos': 'Bosnian', 
                                   'bul': 'Bulgarian', 'cat': 'Catalan', 'chi': 'Chinese', 'cze': 'Czech', 
                                   'dan': 'Danish', 'dut': 'Dutch', 'eng': 'English', 'epo': 'Esperanto', 
                                   'est': 'Estonian', 'fin': 'Finnish', 'fre': 'French', 'geo': 'Georgian', 
                                   'ger': 'German', 'gla': 'Scottish Gaelic', 'gre': 'Greek, Modern', 
                                   'heb': 'Hebrew', 'hin': 'Hindi', 'hrv': 'Croatian', 'hun': 'Hungarian', 
                                   'ice': 'Icelandic', 'ind': 'Indonesian', 'ita': 'Italian', 
                                   'jpn': 'Japanese', 'kin': 'Kinyarwanda', 'kor': 'Korean', 'lat': 'Latin', 
                                   'lav': 'Latvian', 'lit': 'Lithuanian', 'mac': 'Macedonian', 
                                   'mal': 'Malayalam', 'mao': 'Maori', 'may': 'Malay', 
                                   'mul': 'Multiple languages', 'nor': 'Norwegian', 
                                   'per': 'Persian, Iranian', 'pol': 'Polish', 'por': 'Portuguese', 
                                   'pus': 'Pushto', 'rum': 'Romanian, Rumanian, Moldovan', 'rus': 'Russian', 
                                   'san': 'Sanskrit', 'slo': 'Slovak', 'slv': 'Slovenian', 'spa': 'Spanish', 
                                   'srp': 'Serbian',  'swe': 'Swedish', 'tha': 'Thai', 'tur': 'Turkish', 
                                   'ukr': 'Ukrainian', 'und': 'Undetermined', 'vie': 'Vietnamese', 
                                   'wel': 'Welsh'
                                  }
        self.country_names =      [
                                   'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 
                                   'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 
                                   'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 
                                   'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire, Sint Eustatius and Saba', 
                                   'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 
                                   'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 
                                   'Canada', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 
                                   'Christmas Island', 'Cocos Islands', 'Colombia', 'Comoros', 'Democratic Republic of the Congo', 
                                   'Congo', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czechia', 
                                   "Côte d'Ivoire", 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 
                                   'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 
                                   'Falkland Islands', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 
                                   'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 
                                   'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 
                                   'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 
                                   'Holy See', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 
                                   'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 
                                   'Kenya', 'Kiribati', 'North Korea', 'South Korea', 'Kuwait', 'Kyrgyzstan', 
                                   "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 
                                   'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Madagascar', 'Malawi', 'Malaysia', 
                                   'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 
                                   'Mayotte', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 
                                   'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 
                                   'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 
                                   'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 
                                   'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 
                                   'Puerto Rico', 'Qatar', 'Republic of North Macedonia', 'Romania', 'Russian Federation', 'Rwanda', 
                                   'Réunion', 'Saint Barthelemy', 'Saint Helena, Ascension and Tristan da Cunha', 
                                   'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon', 
                                   'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 
                                   'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 
                                   'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 
                                   'South Georgia and the South Sandwich Islands', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 
                                   'Suriname', 'Svalbard and Jan Mayen', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 
                                   'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 
                                   'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 
                                   'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 
                                   'United States Minor Outlying Islands', 'United States of America', 'Uruguay', 'Uzbekistan', 
                                   'Vanuatu', 'Venezuela', 'Viet Nam', 'Virgin Islands (British)', 'Virgin Islands (U.S.)', 
                                   'Wallis and Futuna', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe', 'Aland Islands'
                                  ] 
        self.country_alpha_2 =    [
                                   'AF', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 
                                   'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO', 
                                   'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM', 'CA', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 
                                   'KM', 'CD', 'CG', 'CK', 'CR', 'HR', 'CU', 'CW', 'CY', 'CZ', 'CI', 'DK', 'DJ', 'DM', 'DO', 'EC', 
                                   'EG', 'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 
                                   'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 
                                   'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 
                                   'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 
                                   'LT', 'LU', 'MO', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 
                                   'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 
                                   'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 
                                   'PT', 'PR', 'QA', 'MK', 'RO', 'RU', 'RW', 'RE', 'BL', 'SH', 'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 
                                   'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 
                                   'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 
                                   'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'UM', 'US', 'UY', 'UZ', 'VU', 'VE', 
                                   'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW', 'AX'
                                  ]
        self.country_alpha_3 =    [
                                   'AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATA', 'ATG', 'ARG', 'ARM', 'ABW', 'AUS', 'AUT', 
                                   'AZE', 'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BES', 'BIH', 
                                   'BWA', 'BVT', 'BRA', 'IOT', 'BRN', 'BGR', 'BFA', 'BDI', 'CPV', 'KHM', 'CMR', 'CAN', 'CYM', 'CAF', 
                                   'TCD', 'CHL', 'CHN', 'CXR', 'CCK', 'COL', 'COM', 'COD', 'COG', 'COK', 'CRI', 'HRV', 'CUB', 'CUW', 
                                   'CYP', 'CZE', 'CIV', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 
                                   'ETH', 'FLK', 'FRO', 'FJI', 'FIN', 'FRA', 'GUF', 'PYF', 'ATF', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 
                                   'GIB', 'GRC', 'GRL', 'GRD', 'GLP', 'GUM', 'GTM', 'GGY', 'GIN', 'GNB', 'GUY', 'HTI', 'HMD', 'VAT', 
                                   'HND', 'HKG', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR', 'ITA', 'JAM', 'JPN', 
                                   'JEY', 'JOR', 'KAZ', 'KEN', 'KIR', 'PRK', 'KOR', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', 
                                   'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG', 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MTQ', 'MRT', 
                                   'MUS', 'MYT', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MSR', 'MAR', 'MOZ', 'MMR', 'NAM', 'NRU', 
                                   'NPL', 'NLD', 'NCL', 'NZL', 'NIC', 'NER', 'NGA', 'NIU', 'NFK', 'MNP', 'NOR', 'OMN', 'PAK', 'PLW', 
                                   'PSE', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'PCN', 'POL', 'PRT', 'PRI', 'QAT', 'MKD', 'ROU', 'RUS', 
                                   'RWA', 'REU', 'BLM', 'SHN', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 
                                   'SRB', 'SYC', 'SLE', 'SGP', 'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SGS', 'SSD', 'ESP', 'LKA', 
                                   'SDN', 'SUR', 'SJM', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA', 'THA', 'TLS', 'TGO', 'TKL', 'TON', 
                                   'TTO', 'TUN', 'TUR', 'TKM', 'TCA', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'UMI', 'USA', 'URY', 'UZB', 
                                   'VUT', 'VEN', 'VNM', 'VGB', 'VIR', 'WLF', 'ESH', 'YEM', 'ZMB', 'ZWE', 'ALA'
                                  ]
        self.country_numeric =    [
                                    4, 8, 12, 16, 20, 24, 660, 10, 28, 32, 51, 533, 36, 40, 31, 44, 48, 50, 52, 112, 56, 84, 204, 60, 
                                   64, 68, 535, 70, 72, 74, 76, 86, 96, 100, 854, 108, 132, 116, 120, 124, 136, 140, 148, 152, 156, 
                                   162, 166, 170, 174, 180, 178, 184, 188, 191, 192, 531, 196, 203, 384, 208, 262, 212, 214, 218, 818, 
                                   222, 226, 232, 233, 748, 231, 238, 234, 242, 246, 250, 254, 258, 260, 266, 270, 268, 276, 288, 292, 
                                   300, 304, 308, 312, 316, 320, 831, 324, 624, 328, 332, 334, 336, 340, 344, 348, 352, 356, 360, 364, 
                                   368, 372, 833, 376, 380, 388, 392, 832, 400, 398, 404, 296, 408, 410, 414, 417, 418, 428, 422, 426, 
                                   430, 434, 438, 440, 442, 446, 450, 454, 458, 462, 466, 470, 584, 474, 478, 480, 175, 484, 583, 498, 
                                   492, 496, 499, 500, 504, 508, 104, 516, 520, 524, 528, 540, 554, 558, 562, 566, 570, 574, 580, 578, 
                                   512, 586, 585, 275, 591, 598, 600, 604, 608, 612, 616, 620, 630, 634, 807, 642, 643, 646, 638, 652, 
                                   654, 659, 662, 663, 666, 670, 882, 674, 678, 682, 686, 688, 690, 694, 702, 534, 703, 705, 90, 706, 
                                   710, 239, 728, 724, 144, 729, 740, 744, 752, 756, 760, 158, 762, 834, 764, 626, 768, 772, 776, 780, 
                                   788, 792, 795, 796, 798, 800, 804, 784, 826, 581, 840, 858, 860, 548, 862, 704, 92, 850, 876, 732, 
                                   887, 894, 716, 248
                                  ] 
        self.country_lat_long =   [
                                   (33.93911, 67.709953),    (41.153332, 20.168331),    (28.033886, 1.659626),   (-14.270972, -170.132217), 
                                   (42.546245, 1.601554),    (-11.202692, 17.873887),   (18.220554, -63.068615), (-75.250973, -0.071389), 
                                   (17.060816, -61.796428),  (-38.416097, -63.616672),  (40.069099, 45.038189),  (12.52111, -69.968338), 
                                   (-25.274398, 133.775136), (47.516231, 14.550072),    (40.143105, 47.576927),  (25.03428, -77.39628), 
                                   (25.930414, 50.637772),   (23.684994, 90.356331),    (13.193887, -59.543198), (53.709807, 27.953389), 
                                   (50.503887, 4.469936),    (17.189877, -88.49765),    (9.30769, 2.315834),     (32.321384, -64.75737), 
                                   (27.514162, 90.433601),   (-16.290154, -63.588653),  (12.15, -68.26667),      (43.915886, 17.679076), 
                                   (-22.328474, 24.684866),  (-54.423199, 3.413194),    (-14.235004, -51.92528), (-6.343194, 71.876519), 
                                   (4.535277, 114.727669),   (42.733883, 25.48583),     (12.238333, -1.561593),  (-3.373056, 29.918886), 
                                   (16.002082, -24.013197),  (12.565679, 104.990963),   (7.369722, 12.354722),   (56.130366, -106.346771), 
                                   (19.513469, -80.566956),  (6.611111, 20.939444),     (15.454166, 18.732207),  (-35.675147, -71.542969), 
                                   (35.86166, 104.195397),   (-10.447525, 105.690449),  (-12.164165, 96.870956), (4.570868, -74.297333), 
                                   (-11.875001, 43.872219),  (-4.038333, 21.758664),    (-0.228021, 15.827659),  (-21.236736, -159.777671), 
                                   (9.748917, -83.753428),   (45.1, 15.2),              (21.521757, -77.781167), (12.16957, -68.990021), 
                                   (35.126413, 33.429859),   (49.817492, 15.472962),    (7.539989, -5.54708),    (56.26392, 9.501785), 
                                   (11.825138, 42.590275),   (15.414999, -61.370976),   (18.735693, -70.162651), (-1.831239, -78.183406), 
                                   (26.820553, 30.802498),   (13.794185, -88.89653),    (1.650801, 10.267895),   (15.179384, 39.782334), 
                                   (58.595272, 25.013607),   (-26.522503, 31.465866),   (9.145, 40.489673),      (-51.796253, -59.523613), 
                                   (61.892635, -6.911806),   (-16.578193, 179.414413),  (61.92411, 25.748151),   (46.227638, 2.213749), 
                                   (3.933889, -53.125782),   (-17.679742, -149.406843), (-49.280366, 69.348557), (-0.803689, 11.609444), 
                                   (13.443182, -15.310139),  (42.315407, 43.356892),    (51.165691, 10.451526),  (7.946527, -1.023194), 
                                   (36.137741, -5.345374),   (39.074208, 21.824312),    (71.706936, -42.604303), (12.262776, -61.604171), 
                                   (16.995971, -62.067641),  (13.444304, 144.793731),   (15.783471, -90.230759), (49.465691, -2.585278), 
                                   (9.945587, -9.696645),    (11.803749, -15.180413),   (4.860416, -58.93018),   (18.971187, -72.285215), 
                                   (-53.08181, 73.504158),   (41.902916, 12.453389),    (15.199999, -86.241905), (22.396428, 114.109497), 
                                   (47.162494, 19.503304),   (64.963051, -19.020835),   (20.593684, 78.96288),   (-0.789275, 113.921327), 
                                   (32.427908, 53.688046),   (33.223191, 43.679291),    (53.41291, -8.24389),    (54.236107, -4.548056), 
                                   (31.046051, 34.851612),   (41.87194, 12.56738),      (18.109581, -77.297508), (36.204824, 138.252924), 
                                   (49.214439, -2.13125),    (30.585164, 36.238414),    (48.019573, 66.923684),  (-0.023559, 37.906193), 
                                   (-3.370417, -168.734039), (40.339852, 127.510093),   (35.907757, 127.766922), (29.31166, 47.481766), 
                                   (41.20438, 74.766098),    (19.85627, 102.495496),    (56.879635, 24.603189),  (33.854721, 35.862285), 
                                   (-29.609988, 28.233608),  (6.428055, -9.429499),     (26.3351, 17.228331),    (47.166, 9.555373), 
                                   (55.169438, 23.881275),   (49.815273, 6.129583),     (22.198745, 113.543873), (-18.766947, 46.869107), 
                                   (-13.254308, 34.301525),  (4.210484, 101.975766),    (3.202778, 73.22068),    (17.570692, -3.996166), 
                                   (35.937496, 14.375416),   (7.131474, 171.184478),    (14.641528, -61.024174), (21.00789, -10.940835), 
                                   (-20.348404, 57.552152),  (-12.8275, 45.166244),     (23.634501, -102.552784),(7.425554, 150.550812), 
                                   (47.411631, 28.369885),   (43.750298, 7.412841),     (46.862496, 103.846656), (42.708678, 19.37439), 
                                   (16.742498, -62.187366),  (31.791702, -7.09262),     (-18.665695, 35.529562), (21.913965, 95.956223), 
                                   (-22.95764, 18.49041),    (-0.522778, 166.931503),   (28.394857, 84.124008),  (52.132633, 5.291266), 
                                   (-20.904305, 165.618042), (-40.900557, 174.885971),  (12.865416, -85.207229), (17.607789, 8.081666), 
                                   (9.081999, 8.675277),     (-19.054445, -169.867233), (-29.040835, 167.954712),(17.33083, 145.38469), 
                                   (60.472024, 8.468946),    (21.512583, 55.923255),    (30.375321, 69.345116),  (7.51498, 134.58252), 
                                   (31.952162, 35.233154),   (8.537981, -80.782127),    (-6.314993, 143.95555),  (-23.442503, -58.443832), 
                                   (-9.189967, -75.015152),  (12.879721, 121.774017),   (-24.703615, -127.439308), 
                                   (51.919438, 19.145136),   (39.399872, -8.224454),    (18.220833, -66.590149), (25.354826, 51.183884), 
                                   (41.608635, 21.745275),   (45.943161, 24.96676),     (61.52401, 105.318756),  (-1.940278, 29.873888), 
                                   (-21.115141, 55.536384),  (17.9, 62.8333),           (-24.143474, -10.030696),(17.357822, -62.782998), 
                                   (13.909444, -60.978893),  (18.073099, -63.082199),   (46.941936, -56.27111),  (12.984305, -61.287228), 
                                   (-13.759029, -172.104629),(43.94236, 12.457777),     (0.18636, 6.613081),     (23.885942, 45.079162), 
                                   (14.497401, -14.452362),  (44.016521, 21.005859),    (-4.679574, 55.491977),  (8.460555, -11.779889), 
                                   (1.352083, 103.819836),   (18.0425, 63.0548),        (48.669026, 19.699024),  (46.151241, 14.995463), 
                                   (-9.64571, 160.156194),   (5.152149, 46.199616),     (-30.559482, 22.937506), (-54.429579, -36.587909), 
                                   (6.877, 31.307),          (40.463667, -3.74922),     (7.873054, 80.771797),   (12.862807, 30.217636), 
                                   (3.919305, -56.027783),   (77.553604, 23.670272),    (60.128161, 18.643501),  (46.818188, 8.227512), 
                                   (34.802075, 38.996815),   (23.69781, 120.960515),    (38.861034, 71.276093),  (-6.369028, 34.888822), 
                                   (15.870032, 100.992541),  (-8.874217, 125.727539),   (8.619543, 0.824782),    (-8.967363, -171.855881), 
                                   (-21.178986, -175.198242),(10.691803, -61.222503),   (33.886917, 9.537499),   (38.963745, 35.243322), 
                                   (38.969719, 59.556278),   (21.694025, -71.797928),   (-7.109535, 177.64933),  (1.373333, 32.290275), 
                                   (48.379433, 31.16558),    (23.424076, 53.847818),    (55.378051, -3.435973),  (19.2823, 166.647), 
                                   (37.09024, -95.712891),   (-32.522779, -55.765835),  (41.377491, 64.585262),  (-15.376706, 166.959158), 
                                   (6.42375, -66.58973),     (14.058324, 108.277199),   (18.420695, -64.639968), (18.335765, -64.896335), 
                                   (-13.768752, -177.156097),(24.215527, -12.885834),   (15.552727, 48.516388), 
                                   (-13.133897, 27.849332),  (-19.015438, 29.154857),   (60.1785, 19.9156)
                                  ]
        self.color_names =        [ '#6929c4', '#9f1853', '#198038', '#b28600', '#8a3800', '#1192e8', '#fa4d56', '#002d9c', 
                                    '#009d9a', '#a56eff', '#005d5d', '#570408', '#ee538b', '#012749', '#da1e28', '#f1c21b', 
                                    '#ff832b', '#198038', '#bdd9bf', '#929084', '#ffc857', '#a997df', '#e5323b', '#2e4052', 
                                    '#e1daae', '#ff934f', '#cc2d35', '#214d66', '#848fa2', '#2d3142', '#62a3f0', '#cc5f54', 
                                    '#e6cb60', '#523d02', '#c67ce6', '#00b524', '#4ad9bd', '#f53347', '#565c55',
                                    '#000000', '#ffff00', '#1ce6ff', '#ff34ff', '#ff4a46', '#008941', '#006fa6', '#a30059',
                                    '#ffdbe5', '#7a4900', '#0000a6', '#63ffac', '#b79762', '#004d43', '#8fb0ff', '#997d87',
                                    '#5a0007', '#809693', '#feffe6', '#1b4400', '#4fc601', '#3b5dff', '#4a3b53', '#ff2f80',
                                    '#61615a', '#ba0900', '#6b7900', '#00c2a0', '#ffaa92', '#ff90c9', '#b903aa', '#d16100',
                                    '#ddefff', '#000035', '#7b4f4b', '#a1c299', '#300018', '#0aa6d8', '#013349', '#00846f',
                                    '#372101', '#ffb500', '#c2ffed', '#a079bf', '#cc0744', '#c0b9b2', '#c2ff99', '#001e09',
                                    '#00489c', '#6f0062', '#0cbd66', '#eec3ff', '#456d75', '#b77b68', '#7a87a1', '#788d66',
                                    '#885578', '#fad09f', '#ff8a9a', '#d157a0', '#bec459', '#456648', '#0086ed', '#886f4c',
                                    '#34362d', '#b4a8bd', '#00a6aa', '#452c2c', '#636375', '#a3c8c9', '#ff913f', '#938a81',
                                    '#575329', '#00fecf', '#b05b6f', '#8cd0ff', '#3b9700', '#04f757', '#c8a1a1', '#1e6e00',
                                    '#7900d7', '#a77500', '#6367a9', '#a05837', '#6b002c', '#772600', '#d790ff', '#9b9700',
                                    '#549e79', '#fff69f', '#201625', '#72418f', '#bc23ff', '#99adc0', '#3a2465', '#922329',
                                    '#5b4534', '#fde8dc', '#404e55', '#0089a3', '#cb7e98', '#a4e804', '#324e72', '#6a3a4c',
                                    '#83ab58', '#001c1e', '#d1f7ce', '#004b28', '#c8d0f6', '#a3a489', '#806c66', '#222800',
                                    '#bf5650', '#e83000', '#66796d', '#da007c', '#ff1a59', '#8adbb4', '#1e0200', '#5b4e51',
                                    '#c895c5', '#320033', '#ff6832', '#66e1d3', '#cfcdac', '#d0ac94', '#7ed379', '#012c58',
                                    '#7a7bff', '#d68e01', '#353339', '#78afa1', '#feb2c6', '#75797c', '#837393', '#943a4d',
                                    '#b5f4ff', '#d2dcd5', '#9556bd', '#6a714a', '#001325', '#02525f', '#0aa3f7', '#e98176',
                                    '#dbd5dd', '#5ebcd1', '#3d4f44', '#7e6405', '#02684e', '#962b75', '#8d8546', '#9695c5',
                                    '#e773ce', '#d86a78', '#3e89be', '#ca834e', '#518a87', '#5b113c', '#55813b', '#e704c4',
                                    '#00005f', '#a97399', '#4b8160', '#59738a', '#ff5da7', '#f7c9bf', '#643127', '#513a01',
                                    '#6b94aa', '#51a058', '#a45b02', '#1d1702', '#e20027', '#e7ab63', '#4c6001', '#9c6966',
                                    '#64547b', '#97979e', '#006a66', '#391406', '#f4d749', '#0045d2', '#006c31', '#ddb6d0',
                                    '#7c6571', '#9fb2a4', '#00d891', '#15a08a', '#bc65e9', '#fffffe', '#c6dc99', '#203b3c',
                                    '#671190', '#6b3a64', '#f5e1ff', '#ffa0f2', '#ccaa35', '#374527', '#8bb400', '#797868',
                                    '#c6005a', '#3b000a', '#c86240', '#29607c', '#402334', '#7d5a44', '#ccb87c', '#b88183',
                                    '#aa5199', '#b5d6c3', '#a38469', '#9f94f0', '#a74571', '#b894a6', '#71bb8c', '#00b433',
                                    '#789ec9', '#6d80ba', '#953f00', '#5eff03', '#e4fffc', '#1be177', '#bcb1e5', '#76912f',
                                    '#003109', '#0060cd', '#d20096', '#895563', '#29201d', '#5b3213', '#a76f42', '#89412e',
                                    '#1a3a2a', '#494b5a', '#a88c85', '#f4abaa', '#a3f3ab', '#00c6c8', '#ea8b66', '#958a9f',
                                    '#bdc9d2', '#9fa064', '#be4700', '#658188', '#83a485', '#453c23', '#47675d', '#3a3f00',
                                    '#061203', '#dffb71', '#868e7e', '#98d058', '#6c8f7d', '#d7bfc2', '#3c3e6e', '#d83d66',
                                    '#2f5d9b', '#6c5e46', '#d25b88', '#5b656c', '#00b57f', '#545c46', '#866097', '#365d25',
                                    '#252f99', '#00ccff', '#674e60', '#fc009c', '#92896b', '#1e2324', '#dec9b2', '#9d4948',
                                    '#85abb4', '#342142', '#d09685', '#a4acac', '#00ffff', '#ae9c86', '#742a33', '#0e72c5',
                                    '#afd8ec', '#c064b9', '#91028c', '#feedbf', '#ffb789', '#9cb8e4', '#afffd1', '#2a364c',
                                    '#4f4a43', '#647095', '#34bbff', '#807781', '#920003', '#b3a5a7', '#018615', '#f1ffc8',
                                    '#976f5c', '#ff3bc1', '#ff5f6b', '#077d84', '#f56d93', '#5771da', '#4e1e2a', '#830055',
                                    '#02d346', '#be452d', '#00905e', '#be0028', '#6e96e3', '#007699', '#fec96d', '#9c6a7d',
                                    '#3fa1b8', '#893de3', '#79b4d6', '#7fd4d9', '#6751bb', '#b28d2d', '#e27a05', '#dd9cb8',
                                    '#aabc7a', '#980034', '#561a02', '#8f7f00', '#635000', '#cd7dae', '#8a5e2d', '#ffb3e1',
                                    '#6b6466', '#c6d300', '#0100e2', '#88ec69', '#8fccbe', '#21001c', '#511f4d', '#e3f6e3',
                                    '#ff8eb1', '#6b4f29', '#a37f46', '#6a5950', '#1f2a1a', '#04784d', '#101835', '#e6e0d0',
                                    '#ff74fe', '#00a45f', '#8f5df8', '#4b0059', '#412f23', '#d8939e', '#db9d72', '#604143',
                                    '#b5bace', '#989eb7', '#d2c4db', '#a587af', '#77d796', '#7f8c94', '#ff9b03', '#555196',
                                    '#31ddae', '#74b671', '#802647', '#2a373f', '#014a68', '#696628', '#4c7b6d', '#002c27',
                                    '#7a4522', '#3b5859', '#e5d381', '#fff3ff', '#679fa0', '#261300', '#2c5742', '#9131af',
                                    '#af5d88', '#c7706a', '#61ab1f', '#8cf2d4', '#c5d9b8', '#9ffffb', '#bf45cc', '#493941',
                                    '#863b60', '#b90076', '#003177', '#c582d2', '#c1b394', '#602b70', '#887868', '#babfb0',
                                    '#030012', '#d1acfe', '#7fdefe', '#4b5c71', '#a3a097', '#e66d53', '#637b5d', '#92bea5',
                                    '#00f8b3', '#beddff', '#3db5a7', '#dd3248', '#b6e4de', '#427745', '#598c5a', '#b94c59',
                                    '#8181d5', '#94888b', '#fed6bd', '#536d31', '#6eff92', '#e4e8ff', '#20e200', '#ffd0f2',
                                    '#4c83a1', '#bd7322', '#915c4e', '#8c4787', '#025117', '#a2aa45', '#2d1b21', '#a9ddb0',
                                    '#ff4f78', '#528500', '#009a2e', '#17fce4', '#71555a', '#525d82', '#00195a', '#967874',
                                    '#555558', '#0b212c', '#1e202b', '#efbfc4', '#6f9755', '#6f7586', '#501d1d', '#372d00',
                                    '#741d16', '#5eb393', '#b5b400', '#dd4a38', '#363dff', '#ad6552', '#6635af', '#836bba',
                                    '#98aa7f', '#464836', '#322c3e', '#7cb9ba', '#5b6965', '#707d3d', '#7a001d', '#6e4636',
                                    '#443a38', '#ae81ff', '#489079', '#897334', '#009087', '#da713c', '#361618', '#ff6f01',
                                    '#006679', '#370e77', '#4b3a83', '#c9e2e6', '#c44170', '#ff4526', '#73be54', '#c4df72',
                                    '#adff60', '#00447d', '#dccec9', '#bd9479', '#656e5b', '#ec5200', '#ff6ec2', '#7a617e',
                                    '#ddaea2', '#77837f', '#a53327', '#608eff', '#b599d7', '#a50149', '#4e0025', '#c9b1a9',
                                    '#03919a', '#1b2a25', '#e500f1', '#982e0b', '#b67180', '#e05859', '#006039', '#578f9b',
                                    '#305230', '#ce934c', '#b3c2be', '#c0bac0', '#b506d3', '#170c10', '#4c534f', '#224451',
                                    '#3e4141', '#78726d', '#b6602b', '#200441', '#ddb588', '#497200', '#c5aab6', '#033c61',
                                    '#71b2f5', '#a9e088', '#4979b0', '#a2c3df', '#784149', '#2d2b17', '#3e0e2f', '#57344c',
                                    '#0091be', '#e451d1', '#4b4b6a', '#5c011a', '#7c8060', '#ff9491', '#4c325d', '#005c8b',
                                    '#e5fda4', '#68d1b6', '#032641', '#140023', '#8683a9', '#cfff00', '#a72c3e', '#34475a',
                                    '#b1bb9a', '#b4a04f', '#8d918e', '#a168a6', '#813d3a', '#425218', '#da8386', '#776133',
                                    '#563930', '#8498ae', '#90c1d3', '#b5666b', '#9b585e', '#856465', '#ad7c90', '#e2bc00',
                                    '#e3aae0', '#b2c2fe', '#fd0039', '#009b75', '#fff46d', '#e87eac', '#dfe3e6', '#848590',
                                    '#aa9297', '#83a193', '#577977', '#3e7158', '#c64289', '#ea0072', '#c4a8cb', '#55c899',
                                    '#e78fcf', '#004547', '#f6e2e3', '#966716', '#378fdb', '#435e6a', '#da0004', '#1b000f',
                                    '#5b9c8f', '#6e2b52', '#011115', '#e3e8c4', '#ae3b85', '#ea1ca9', '#ff9e6b', '#457d8b',
                                    '#92678b', '#00cdbb', '#9ccc04', '#002e38', '#96c57f', '#cff6b4', '#492818', '#766e52',
                                    '#20370e', '#e3d19f', '#2e3c30', '#b2eace', '#f3bda4', '#a24e3d', '#976fd9', '#8c9fa8',
                                    '#7c2b73', '#4e5f37', '#5d5462', '#90956f', '#6aa776', '#dbcbf6', '#da71ff', '#987c95',
                                    '#52323c', '#bb3c42', '#584d39', '#4fc15f', '#a2b9c1', '#79db21', '#1d5958', '#bd744e',
                                    '#160b00', '#20221a', '#6b8295', '#00e0e4', '#102401', '#1b782a', '#daa9b5', '#b0415d',
                                    '#859253', '#97a094', '#06e3c4', '#47688c', '#7c6755', '#075c00', '#7560d5', '#7d9f00',
                                    '#c36d96', '#4d913e', '#5f4276', '#fce4c8', '#303052', '#4f381b', '#e5a532', '#706690',
                                    '#aa9a92', '#237363', '#73013e', '#ff9079', '#a79a74', '#029bdb', '#ff0169', '#c7d2e7',
                                    '#ca8869', '#80ffcd', '#bb1f69', '#90b0ab', '#7d74a9', '#fcc7db', '#99375b', '#00ab4d',
                                    '#abaed1', '#be9d91', '#e6e5a7', '#332c22', '#dd587b', '#f5fff7', '#5d3033', '#6d3800',
                                    '#ff0020', '#b57bb3', '#d7ffe6', '#c535a9', '#260009', '#6a8781', '#a8abb4', '#d45262',
                                    '#794b61', '#4621b2', '#8da4db', '#c7c890', '#6fe9ad', '#a243a7', '#b2b081', '#181b00',
                                    '#286154', '#4ca43b', '#6a9573', '#a8441d', '#5c727b', '#738671', '#d0cfcb', '#897b77',
                                    '#1f3f22', '#4145a7', '#da9894', '#a1757a', '#63243c', '#adaaff', '#00cde2', '#ddbc62',
                                    '#698eb1', '#208462', '#00b7e0', '#614a44', '#9bbb57', '#7a5c54', '#857a50', '#766b7e',
                                    '#014833', '#ff8347', '#7a8eba', '#274740', '#946444', '#ebd8e6', '#646241', '#373917',
                                    '#6ad450', '#81817b', '#d499e3', '#979440', '#011a12', '#526554', '#b5885c', '#a499a5',
                                    '#03ad89', '#b3008b', '#e3c4b5', '#96531f', '#867175', '#74569e', '#617d9f', '#e70452',
                                    '#067eaf', '#a697b6', '#b787a8', '#9cff93', '#311d19', '#3a9459', '#6e746e', '#b0c5ae',
                                    '#84edf7', '#ed3488', '#754c78', '#384644', '#c7847b', '#00b6c5', '#7fa670', '#c1af9e',
                                    '#2a7fff', '#72a58c', '#ffc07f', '#9debdd', '#d97c8e', '#7e7c93', '#62e674', '#b5639e',
                                    '#ffa861', '#c2a580', '#8d9c83', '#b70546', '#372b2e', '#0098ff', '#985975', '#20204c',
                                    '#ff6c60', '#445083', '#8502aa', '#72361f', '#9676a3', '#484449', '#ced6c2', '#3b164a',
                                    '#cca763', '#2c7f77', '#02227b', '#a37e6f', '#cde6dc', '#cdfffb', '#be811a', '#f77183',
                                    '#ede6e2', '#cdc6b4', '#ffe09e', '#3a7271', '#ff7b59', '#4e4e01', '#4ac684', '#8bc891',
                                    '#bc8a96', '#cf6353', '#dcde5c', '#5eaadd', '#f6a0ad', '#e269aa', '#a3dae4', '#436e83',
                                    '#002e17', '#ecfbff', '#a1c2b6', '#50003f', '#71695b', '#67c4bb', '#536eff', '#5d5a48',
                                    '#890039', '#969381', '#371521', '#5e4665', '#aa62c3', '#8d6f81', '#2c6135', '#410601',
                                    '#564620', '#e69034', '#6da6bd', '#e58e56', '#e3a68b', '#48b176', '#d27d67', '#b5b268',
                                    '#7f8427', '#ff84e6', '#435740', '#eae408', '#f4f5ff', '#325800', '#4b6ba5', '#adceff',
                                    '#9b8acc', '#885138', '#5875c1', '#7e7311', '#fea5ca', '#9f8b5b', '#a55b54', '#89006a',
                                    '#af756f', '#2a2000', '#576e4a', '#7f9eff', '#7499a1', '#ffb550', '#00011e', '#d1511c',
                                    '#688151', '#bc908a', '#78c8eb', '#8502ff', '#483d30', '#c42221', '#5ea7ff', '#785715',
                                    '#0cea91', '#fffaed', '#b3af9d', '#3e3d52', '#5a9bc2', '#9c2f90', '#8d5700', '#add79c',
                                    '#00768b', '#337d00', '#c59700', '#3156dc', '#944575', '#ecffdc', '#d24cb2', '#97703c',
                                    '#4c257f', '#9e0366', '#88ffec', '#b56481', '#396d2b', '#56735f', '#988376', '#9bb195',
                                    '#a9795c', '#e4c5d3', '#9f4f67', '#1e2b39', '#664327', '#afce78', '#322edf', '#86b487',
                                    '#c23000', '#abe86b', '#96656d', '#250e35', '#a60019', '#0080cf', '#caefff', '#323f61',
                                    '#a449dc', '#6a9d3b', '#ff5ae4', '#636a01', '#d16cda', '#736060', '#ffbaad', '#d369b4',
                                    '#ffded6', '#6c6d74', '#927d5e', '#845d70', '#5b62c1', '#2f4a36', '#e45f35', '#ff3b53',
                                    '#ac84dd', '#762988', '#70ec98', '#408543', '#2c3533', '#2e182d', '#323925', '#19181b',
                                    '#2f2e2c', '#023c32', '#9b9ee2', '#58afad', '#5c424d', '#7ac5a6', '#685d75', '#b9bcbd',
                                    '#834357', '#1a7b42', '#2e57aa', '#e55199', '#316e47', '#cd00c5', '#6a004d', '#7fbbec',
                                    '#f35691', '#d7c54a', '#62acb7', '#cba1bc', '#a28a9a', '#6c3f3b', '#ffe47d', '#dcbae3',
                                    '#5f816d', '#3a404a', '#7dbf32', '#e6ecdc', '#852c19', '#285366', '#b8cb9c', '#0e0d00',
                                    '#4b5d56', '#6b543f', '#e27172', '#0568ec', '#2eb500', '#d21656', '#efafff', '#682021',
                                    '#2d2011', '#da4cff', '#70968e', '#ff7b7d', '#4a1930', '#e8c282', '#e7dbbc', '#a68486',
                                    '#1f263c', '#36574e', '#52ce79', '#adaaa9', '#8a9f45', '#6542d2', '#00fb8c', '#5d697b',
                                    '#ccd27f', '#94a5a1', '#790229', '#e383e6', '#7ea4c1', '#4e4452', '#4b2c00', '#620b70',
                                    '#314c1e', '#874aa6', '#e30091', '#66460a', '#eb9a8b', '#eac3a3', '#98eab3', '#ab9180',
                                    '#b8552f', '#1a2b2f', '#94ddc5', '#9d8c76', '#9c8333', '#94a9c9', '#392935', '#8c675e',
                                    '#cce93a', '#917100', '#01400b', '#449896', '#1ca370', '#e08da7', '#8b4a4e', '#667776',
                                    '#4692ad', '#67bda8', '#69255c', '#d3bfff', '#4a5132', '#7e9285', '#77733c', '#e7a0cc',
                                    '#51a288', '#2c656a', '#4d5c5e', '#c9403a', '#ddd7f3', '#005844', '#b4a200', '#488f69',
                                    '#858182', '#d4e9b9', '#3d7397', '#cae8ce', '#d60034', '#aa6746', '#9e5585', '#ba6200',
                                    '#dee3E9', '#ebbaB5', '#fef3c7', '#a6e3d7', '#cbb4d5', '#808b96', '#f7dc6f', '#48c9b0',
                                    '#af7ac5', '#ec7063', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53',
                                    '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16',
                                    '#d85679', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80',
                                    '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'
                                  ]
        self.data, self.entries = self.__read_bib(file_bib, db, del_duplicated)
        self.__make_bib()
    
    # Function: Prepare .bib File
    def __make_bib(self, verbose = True):
        self.ask_gpt_ap            = -1
        self.ask_gpt_cp            = -1
        self.ask_gpt_ip            = -1
        self.ask_gpt_sp            = -1
        self.ask_gpt_bp            = -1
        self.ask_gpt_ct            = -1
        self.ask_gpt_ep            = -1
        self.ask_gpt_ng            = -1
        self.ask_gpt_rt            = -1
        self.ask_gpt_sk            = -1
        self.ask_gpt_wd            = -1
        self.author_country_map    = -1
        self.corr_a_country_map    = -1
        self.frst_a_country_map    = -1
        self.author_inst_map       = -1
        self.corr_a_inst_map       = -1
        self.frst_a_inst_map       = -1  
        self.top_y_x               = -1
        self.heat_y_x              = -1
        self.top_refs              = -1
        self.rpys_pk               = -1
        self.rpys_rs               = -1
        self.top_co_c              = -1
        self.data['year']          = self.data['year'].replace('UNKNOWN', '0')
        self.dy                    = pd.to_numeric(self.data['year'], downcast = 'float')
        self.date_str              = int(self.dy.min())
        self.date_end              = int(self.dy.max())
        self.doc_types             = self.data['document_type'].value_counts().sort_index()
        self.av_d_year             = self.dy.value_counts().sort_index()
        self.av_d_year             = round(self.av_d_year.mean(), 2)
        self.citation              = self.__get_citations(self.data['note'])
        self.av_c_doc              = round(sum(self.citation)/self.data.shape[0], 2)
        self.ref, self.u_ref       = self.__get_str(entry = 'references', s = ';',     lower = False, sorting = True)
        self.aut, self.u_aut       = self.__get_str(entry = 'author',     s = ' and ', lower = True,  sorting = True)
        self.aut_h                 = self.h_index()
        self.aut_g                 = self.g_index()
        self.aut_e                 = self.e_index()
        self.aut_docs              = [len(item) for item in self.aut]
        self.aut_single            = len([item  for item in self.aut_docs if item == 1])
        self.aut_multi             = [item for item in self.aut_docs if item > 1]
        self.aut_cit               = self.__get_counts(self.u_aut, self.aut, self.citation)
        self.author_to_papers      = defaultdict(list)
        for paper_idx, authors in enumerate(self.aut):
            for author in authors:
                self.author_to_papers[author].append(paper_idx)
        self.kid, self.u_kid       = self.__get_str(entry = 'keywords', s = ';', lower = True, sorting = True)
        self.u_kid, self.kid_count = self.filter_list(u_e = self.u_kid, e = self.kid)
        self.auk, self.u_auk       = self.__get_str(entry = 'author_keywords', s = ';', lower = True, sorting = True)
        self.u_auk, self.auk_count = self.filter_list(u_e = self.u_auk, e = self.auk)
        self.jou, self.u_jou       = self.__get_str(entry = 'abbrev_source_title', s = ';', lower = True, sorting = True)
        self.u_jou, self.jou_count = self.filter_list(u_e = self.u_jou, e = self.jou)
        self.jou_cit               = self.__get_counts(self.u_jou, self.jou, self.citation)
        self.lan, self.u_lan       = self.__get_str(entry = 'language', s = '.', lower = True, sorting = True)
        self.u_lan, self.lan_count = self.filter_list(u_e = self.u_lan, e = self.lan, simple = True)
        self.ctr, self.u_ctr       = self.__get_countries()
        self.ctr                   = self.replace_unknowns(self.ctr)
        self.u_ctr, self.ctr_count = self.filter_list(u_e = self.u_ctr, e = self.ctr, simple = True)
        self.ctr_cit               = self.__get_counts(self.u_ctr, self.ctr, self.citation)
        self.uni, self.u_uni       = self.__get_institutions() 
        self.uni                   = self.replace_unknowns(self.uni)
        self.u_uni, self.uni_count = self.filter_list(u_e = self.u_uni, e = self.uni, simple = True)
        self.uni_cit               = self.__get_counts(self.u_uni, self.uni, self.citation)
        self.doc_aut               = self.__get_counts(self.u_aut, self.aut)
        self.av_doc_aut            = round(sum(self.doc_aut)/len(self.doc_aut), 2)
        self.t_c, self.s_c         = self.__total_and_self_citations()
        self.r_c                   = [self.s_c[i]/max(self.t_c[i], 1) for i in range(0, len(self.t_c))]
        self.natsort               = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]  
        self.dy_c_year             = self.__get_collaboration_year()
        self.u_ref                 = [ref for ref in self.u_ref if ref.lower() != 'unknown']
        self.dy_ref                = self.__get_ref_year()
        self.u_ref_id              = self.__get_ref_id()
        ref_map                    = dict(zip(self.u_ref, self.u_ref_id))
        self.ref_id                = []
        for ref_list in self.ref:
            r_id = [ref_map.get(ref, ref) for ref in ref_list]
            self.ref_id.append(r_id)
        self.__id_document()
        self.__id_author()
        self.__id_source()
        self.__id_institution()
        self.__id_country()
        self.__id_kwa()
        self.__id_kwp()
        if (verbose == True):
            for i in range(0, len(self.vb)):
                print(self.vb[i])
        return
    
    # Function: Document ID
    def __id_document(self):
        doc_list          = [str(i) for i in range(0, self.data.shape[0])]
        docs              = [self.data.loc[i, 'author']+' ('+self.data.loc[i, 'year']+'). '+self.data.loc[i, 'title']+'. '+self.data.loc[i, 'journal']+'. doi:'+self.data.loc[i, 'doi']+'. ' for i in range(0, self.data.shape[0])]
        self.table_id_doc = pd.DataFrame(zip(doc_list, docs), columns = ['ID', 'Document'])
        self.dict_id_doc  = dict(zip(doc_list, docs))
        return
    
    # Function: Author ID
    def __id_author(self):
        aut_list          = ['a_'+str(i) for i in range(0, len(self.u_aut))]
        self.table_id_aut = pd.DataFrame(zip(aut_list, self.u_aut), columns = ['ID', 'Author'])
        self.dict_id_aut  = dict(zip(aut_list, self.u_aut))
        self.dict_aut_id  = dict(zip(self.u_aut, aut_list))
        return
    
    # Function: Source ID
    def __id_source(self):
        jou_list          = ['j_'+str(i) for i in range(0, len(self.u_jou))]
        self.table_id_jou = pd.DataFrame(zip(jou_list, self.u_jou), columns = ['ID', 'Source'])
        self.dict_id_jou  = dict(zip(jou_list, self.u_jou))
        self.dict_jou_id  = dict(zip(self.u_jou, jou_list))
        return
    
    # Function: Institution ID
    def __id_institution(self):
        uni_list          = ['i_'+str(i) for i in range(0, len(self.u_uni))]
        self.table_id_uni = pd.DataFrame(zip(uni_list, self.u_uni), columns = ['ID', 'Institution'])
        self.dict_id_uni  = dict(zip(uni_list, self.u_uni))
        self.dict_uni_id  = dict(zip(self.u_uni, uni_list))
        return
    
    # Function: Country ID
    def __id_country(self):
        ctr_list          = ['c_'+str(i) for i in range(0, len(self.u_ctr))]
        self.table_id_ctr = pd.DataFrame(zip(ctr_list, self.u_ctr), columns = ['ID', 'Country'])
        self.dict_id_ctr  = dict(zip(ctr_list, self.u_ctr))
        self.dict_ctr_id  = dict(zip(self.u_ctr, ctr_list))
        return
    
    # Function: Authors' Keyword ID
    def __id_kwa(self):
        kwa_list          = ['k_'+str(i) for i in range(0, len(self.u_auk))]
        self.table_id_kwa = pd.DataFrame(zip(kwa_list, self.u_auk), columns = ['ID', 'KWA'])
        self.dict_id_kwa  = dict(zip(kwa_list, self.u_auk))
        self.dict_kwa_id  = dict(zip(self.u_auk, kwa_list))
        return
    
    # Function: Keywords Plus ID
    def __id_kwp(self):
        kwp_list          = ['p_'+str(i) for i in range(0, len(self.u_kid))]
        self.table_id_kwp = pd.DataFrame(zip(kwp_list, self.u_kid), columns = ['ID', 'KWP'])
        self.dict_id_kwp  = dict(zip(kwp_list, self.u_kid))
        self.dict_kwp_id  = dict(zip(self.u_kid, kwp_list))
        return
    
    # Function: ID types
    def id_doc_types(self):
        dt     = self.doc_types.index.to_list()
        dt_ids = []
        for i in range(0, len(dt)):
            item = dt[i]
            idx  = self.data.index[self.data['document_type'] == item].tolist()
            dt_ids.append([item, idx])
        report_dt = pd.DataFrame(dt_ids, columns = ['Document Types', 'IDs'])
        return report_dt

    # Function: Filter Lists
    def filter_list(self, u_e = [], e = [], simple = False):
        if (simple == True):
            e_      = [item for sublist in e for item in sublist]
            e_count = [e_.count(item) for item in u_e]   
        else:
            u_e     = [item for item in u_e if item.lower() != 'unknown']
            e_      = [item for sublist in e for item in sublist]
            e_count = [e_.count(item) for item in u_e]
            idx     = sorted(range(len(e_count)), key = e_count.__getitem__)
            idx.reverse()
            u_e     = [u_e[i]     for i in idx]
            e_count = [e_count[i] for i in idx]
        return u_e, e_count

    # Function: Filter Bib
    def filter_bib(self, documents = [], doc_type = [], year_str = -1, year_end = -1, sources = [], core = -1, country = [], language = [], abstract = False):
        docs = []
        if (len(documents) > 0):
            self.data = self.data.iloc[documents, :]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(doc_type) > 0):
            for item in doc_type:
                if (sum(self.data['document_type'].isin([item])) > 0):
                    docs.append(item) 
            self.data = self.data[self.data['document_type'].isin(docs)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (year_str > -1):
            self.data = self.data[self.data['year'] >= str(year_str)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (year_end > -1):
            self.data = self.data[self.data['year'] <= str(year_end)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(sources) > 0):
            src_idx = []
            for source in sources:
                for i in range(0, len(self.jou)):
                    if (source == self.jou[i][0]):
                        src_idx.append(i)
            if (len(src_idx) > 0):
                self.data = self.data.iloc[src_idx, :]
                self.data = self.data.reset_index(drop = True)
                self.__make_bib(verbose = False)
        if (core == 1 or core == 2 or core == 3 or core == 12 or core == 23):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            c2    = int(value[-1]*(2/3))
            if (core ==  1):
                key = [key[i] for i in range(0, len(key)) if value[i] <= c1]
            if (core ==  2):
                key = [key[i] for i in range(0, len(key)) if value[i] > c1 and value[i] <= c2]
            if (core ==  3):
               key = [key[i] for i in range(0, len(key)) if value[i] > c2]
            if (core == 12):
                key = [key[i] for i in range(0, len(key)) if value[i] <= c2]
            if (core == 23):
                key = [key[i] for i in range(0, len(key)) if value[i] > c1]
            sources   = self.data['abbrev_source_title'].str.lower()
            self.data = self.data[sources.isin(key)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(country) > 0):
            ctr_idx   = [i for i in range(0, len(self.ctr)) if any(x in country for x in self.ctr[i])] 
            if (len(ctr_idx) > 0):
                self.data = self.data.iloc[ctr_idx, :]
                self.data = self.data.reset_index(drop = True)
                self.__make_bib(verbose = False)
        if (len(language) > 0):
            self.data = self.data[self.data['language'].isin(language)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (abstract == True):
            self.data = self.data[self.data['abstract'] != 'UNKNOWN']
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        self.__update_vb()
        self.__make_bib(verbose = True)
        return
        
    # Function: Clean DOI entries
    def clean_doi(doi):
        valid_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ./-_:')
        cleaned_doi = ''
        for char in doi:
            if char in valid_chars:
                cleaned_doi = cleaned_doi + char
            else:
                break
        return cleaned_doi
        
    # Function: Fuzzy String Matcher # Entry = self.u_aut, self.u_inst, or list with Unique items
    def fuzzy_matcher(self, entry, tgt = [], cut_ratio = 0.80):
        u_lst   = [item for item in entry]
        matches = {item: [] for item in u_lst}
        if (tgt):
            for target in tgt:
                if (target not in u_lst):
                    continue  
                for other in u_lst:
                    if (other == target):
                        continue  
                    ratio = SequenceMatcher(None, target, other).ratio()
                    #ratio = fuzz.ratio(target, other)/100
                    if (cut_ratio <= ratio < 1):
                        matches[target].append(other)
                        matches[other].append(target)
        else:
            for i, j in combinations(range(0, len(u_lst)), 2):
                str1, str2 = u_lst[i], u_lst[j]
                ratio      = SequenceMatcher(None, str1, str2).ratio()
                #ratio      = fuzz.ratio(str1, str2)/100
                if (cut_ratio <= ratio < 1):
                    matches[str1].append(str2)
                    matches[str2].append(str1)
        matches = {k: v for k, v in matches.items() if v}
        return matches
    
    # Function: Merge Database
    def merge_database(self, file_bib, db, del_duplicated):
        old_vb   = [item for item in self.vb]
        old_size = self.data.shape[0]
        print('############################################################################')
        print('')
        print('Original Database')
        print('')
        for i in range(0, len(old_vb)):
            print(old_vb[i])
        print('')
        print('############################################################################')
        print('')
        print('Added Database')
        print('')
        data, _    = self.__read_bib(file_bib, db, del_duplicated)
        self.data  = pd.concat([self.data, data]) 
        self.data  = self.data.reset_index(drop = True)
        self.data  = self.data.fillna('UNKNOWN')
        duplicated = self.data['doi'].duplicated()
        title      = self.data['title']
        title      = title.to_list()
        title      = self.clear_text(title, stop_words  = [], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [])
        t_dupl     = pd.Series(title).duplicated()
        for i in range(0, duplicated.shape[0]):
            if (self.data.loc[i, 'doi'] == 'UNKNOWN'):
                duplicated[i] = False
            if (t_dupl[i] == True):
                duplicated[i] = True
        idx        = list(duplicated.index[duplicated])
        self.data.drop(idx, axis = 0, inplace = True)
        self.data  = self.data.reset_index(drop = True)
        size       = self.data.shape[0]
        self.__make_bib(verbose = True)
        dt         = self.data['document_type'].value_counts()
        dt         = dt.sort_index(axis = 0)
        self.vb    = []
        print('')
        print('############################################################################')
        print('')
        print('Merging Information:')
        print('')
        print( 'A Total of ' + str(size) + ' Documents were Found ( ' + str(size - old_size) + ' New Documents from the Added Database )')
        self.vb.append('A Total of ' + str(size) + ' Documents were Found')
        print('')
        for i in range(0, dt.shape[0]):
            print(dt.index[i], ' = ', dt[i])
            self.vb.append(dt.index[i] + ' = ' + str(dt[i]))
        print('')
        print('############################################################################')
        return

    # Function: Save Working Database
    def save_database(self, sep = '\t', name = 'data.csv'):
        self.data.to_csv(name, index = False)
        return
    
    # Function: Load Working Database
    def load_database(self, name = 'data.csv'):
        data      = pd.read_csv(name, dtype = str)
        self.data = data.copy(deep = True)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Author
    def merge_author(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'author'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'author'] = target.replace(name.lower(), replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Institution
    def merge_institution(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                if (self.data.loc[i, 'source'].lower() == 'scopus' or self.data.loc[i, 'source'].lower() == 'pubmed'):
                    target = self.data.loc[i, 'affiliation'].lower()
                elif (self.data.loc[i, 'source'].lower() == 'wos'):
                    target = self.data.loc[i, 'affiliation_'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'affiliation'] = target.replace(name, replace_for)
        self.__make_bib(verbose = False)
        return
                
    # Function: Merge Country
    def merge_country(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                if (self.data.loc[i, 'source'].lower() == 'scopus' or self.data.loc[i, 'source'].lower() == 'pubmed'):
                    target = self.data.loc[i, 'affiliation'].lower()
                elif (self.data.loc[i, 'source'].lower() == 'wos'):
                    target = self.data.loc[i, 'affiliation_'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'affiliation'] = target.replace(name.lower(), replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Language
    def merge_language(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'language'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'language'] = target.replace(name.lower(), replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Source
    def merge_source(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'abbrev_source_title'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'abbrev_source_title'] = target.replace(name.lower(), replace_for)
        self.__make_bib(verbose = False)
        return

    # Function: Merge Reference
    def merge_reference(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'references'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'references'] = target.replace(name.lower(), replace_for)
        self.__make_bib(verbose = False)
        return

    # Function: Replace Keyword Plus
    def replace_keyword_plus(self, replace_all, edit = True):
        if edit:
            result_strings        = ['; '.join([phrase for (phrase, _) in sub]) for sub in replace_all]
            self.data['keywords'] = result_strings 
        else:
            self.data['keywords'] = replace_all
        self.__make_bib(verbose = False)
        return
    
    # Function: Transform Hex to RGBa
    def __hex_rgba(self, hxc = '#ba6200', alpha = 0.15):
        if (hxc.find('#') == 0):
            hxc  = hxc.lstrip('#')
            rgb  = tuple(int(hxc[i:i+2], 16) for i in (0, 2, 4))
            rgba = 'rgba('+str(rgb[0])+','+str(rgb[1])+','+str(rgb[2])+','+str(alpha)+')'
        else:
            rgba = 'black'
        return rgba
    
    #############################################################################
    
    # Function: EDA .bib docs
    def eda_bib(self):
        report = []
        report.append(['Timespan', str(self.date_str)+'-'+str(self.date_end)])
        report.append(['Total Number of Countries', len(self.u_ctr)])
        report.append(['Total Number of Institutions', len(self.u_uni)])
        report.append(['Total Number of Sources', len(self.u_jou)])
        report.append(['Total Number of References', len(self.u_ref)])
        report.append(['Total Number of Languages', len(self.u_lan)])
        for i in range(0, len(self.u_lan)):
            report.append(['--'+self.u_lan[i]+' (# of docs)', self.lan_count[i]])
        report.append(['-//-', '-//-'])
        report.append(['Total Number of Documents', self.data.shape[0]])
        for i in range(0, self.doc_types.shape[0]):
            report.append(['--'+self.doc_types.index[i], self.doc_types[i]])
        report.append(['Average Documents per Author',self. av_doc_aut])
        report.append(['Average Documents per Institution', round(sum(self.uni_count)/len(self.uni_count), 2) if len(self.uni_count) > 0 else 0])
        report.append(['Average Documents per Source', round(sum(self.jou_count)/len(self.jou_count), 2) if len(self.jou_count) > 0 else 0])
        report.append(['Average Documents per Year', self.av_d_year])
        report.append(['-//-', '-//-'])
        report.append(['Total Number of Authors', len(self.u_aut)])
        report.append(['Total Number of Authors Keywords', len(self.u_auk)])
        report.append(['Total Number of Authors Keywords Plus', len(self.u_kid)])
        report.append(['Total Single-Authored Documents', self.aut_single])
        report.append(['Total Multi-Authored Documents', len(self.aut_multi)])
        report.append(['Average Collaboration Index', self.dy_c_year.iloc[-1, -1]])
        report.append(['Max H-Index', max(self.aut_h)])
        report.append(['-//-', '-//-'])
        report.append(['Total Number of Citations', sum(self.citation)])
        report.append(['Average Citations per Author', round(sum(self.citation)/len(self.u_aut), 2) if len(self.u_aut) > 0 else 0])
        report.append(['Average Citations per Institution', round(sum(self.citation)/len(self.u_uni), 2) if len(self.u_uni) > 0 else 0])
        report.append(['Average Citations per Document', self.av_c_doc])
        report.append(['Average Citations per Source', round(sum(self.jou_cit)/len(self.jou_cit), 2) if len(self.jou_cit) > 0 else 0])
        report.append(['-//-', '-//-'])
        self.ask_gpt_rt = pd.DataFrame(report, columns = ['Main Information', 'Results'])
        report_df       = pd.DataFrame(report, columns = ['Main Information', 'Results'])
        return report_df

    # Function: Health of .bib docs 
    def health_bib(self):
        n      = self.data.shape[0]
        health = []
        check  = {
                    'abbrev_source_title': 'Sources',
                    'abstract':            'Abstracts',
                    'affiliation':         'Affiliation',
                    'author':              'Author(s)',
                    'doi':                 'DOI',
                    'author_keywords':     'Keywords - Authors',
                    'keywords':            'Keywords - Plus',
                    'references':          'References',
                    'year':                'Year'
                  }
        for col, name in check.items():
            if (col in self.data.columns):
                unknown_count = (self.data[col].astype(str) == 'UNKNOWN').sum()
                health_metric = ((n - unknown_count) / n) * 100
                health.append([name, f'{health_metric:.2f}%', str(n - unknown_count)])
            else:
                health.append([name, None])
        health_df = pd.DataFrame(health, columns = ['Entries', 'Completeness (%)', 'Number of  Docs'])
        return health_df
    
    ##############################################################################

    # Function: Read .bib File
    def __read_bib(self, bib, db = 'scopus', del_duplicated = True):
        
        #----------------------------------------------------------------------
        
        def assign_authors_to_affiliations(authors_str, affiliations_str):
            authors          = [a.strip() for a in authors_str.split(' and ')]
            affiliations     = [a.strip() for a in affiliations_str.split(';')]
            new_affiliations = []
            for i, aff in enumerate(affiliations):
                if i < len(authors):
                    new_affiliations.append(f"{authors[i]} {aff}")
                else:
                    new_affiliations.append(aff)
            return '; '.join(new_affiliations)
        
        def get_corresponding_authors_and_affiliations(corr_address):
            if not isinstance(corr_address, str):
                return [], []
            match = re.search(r'Corresponding Author\s+([^;]+);([^;]+)', corr_address, re.IGNORECASE)
            if not match:
                return [], []
            authors_part      = match.group(1).strip()
            affiliations_part = match.group(2).strip()
            authors           = [a for a in re.split(r'\s+and\s+|;', authors_part)]
            affiliations      = [affiliations_part]
            return authors, affiliations
        
        def map_authors_to_affiliations(row):
            authors_str             = row['author']
            affiliations_str        = row['affiliation']
            correspondence_address1 = row['correspondence_address1']
            if isinstance(authors_str, str):
                authors = [a for a in re.split(r'\s+and\s+|;', authors_str)]
            else:
                authors = []
            if isinstance(affiliations_str, str):
                affiliations = [a.strip() for a in affiliations_str.split(';')]
            else:
                affiliations = []
            ca_authors, ca_affiliations = get_corresponding_authors_and_affiliations(correspondence_address1)
            new_affiliations            = []
            for ca_author, ca_aff in zip(ca_authors, ca_affiliations):
                new_affiliations.append(f"{ca_author} {ca_aff}")
            for ca_aff in ca_affiliations:
                affiliations = [aff for aff in affiliations if ca_aff.lower() not in aff.lower()]
            remaining_authors = [a for a in authors if a not in ca_authors]
            for i, author in enumerate(remaining_authors):
                if i < len(affiliations):
                    aff = affiliations[i]
                    new_affiliations.append(f"{author} {aff}")
            transformed_affiliation = '; '.join(new_affiliations)
            return transformed_affiliation

        #----------------------------------------------------------------------
        
        self.vb        = []
        db             = db.lower()
        file_extension = os.path.splitext(bib)[1].lower()
        f_list         = []
        if  (db == 'scopus' and file_extension == '.csv'):
            #data         = pd.read_csv(bib, encoding = 'utf8', dtype = str)
            try:
                data = pd.read_csv(bib, encoding = 'utf8', dtype = str)
            except UnicodeDecodeError:
                data = pd.read_csv(bib, encoding = 'cp1252', dtype = str)
            data.columns = data.columns.str.lower()
            if ('abbrev_source_title' not in data.columns and 'abbreviated source title' in data.columns):
                data.rename(columns = {'abbreviated source title': 'abbrev_source_title'}, inplace = True)
            if ('abbrev_source_title' not in data.columns and 'journal' in data.columns):
                data.rename(columns = {'journal': 'abbrev_source_title'}, inplace = True)
            if ('document_type' not in data.columns and 'document type' in data.columns):
                data.rename(columns = {'document type': 'document_type'}, inplace = True)
            if ('art_number.' not in data.columns and 'art. no.' in data.columns):
                data.rename(columns = {'art. no.': 'art_number'}, inplace = True)
            if ('author_keywords' not in data.columns and 'author keywords' in data.columns):
                data.rename(columns = {'author keywords': 'author_keywords'}, inplace = True)
            if ('author' not in data.columns and 'authors' in data.columns):
                data.rename(columns = {'authors': 'author'}, inplace = True)
            if ('chemicals_cas' not in data.columns and r'chemicals/cas' in data.columns):
                data.rename(columns = {r'chemicals/cas': 'chemicals_cas'}, inplace = True)
            if ('correspondence_address' not in data.columns and 'correspondence address' in data.columns):
                data.rename(columns = {'correspondence address': 'correspondence_address'}, inplace = True)
            if ('editor' not in data.columns and 'editors' in data.columns):
                data.rename(columns = {'editors': 'editor'}, inplace = True)
            if ('funding_details' not in data.columns and 'funding details' in data.columns):
                data.rename(columns = {'funding details': 'funding_details'}, inplace = True)
            if ('keywords' not in data.columns and 'index keywords' in data.columns):
                data.rename(columns = {'index keywords': 'keywords'}, inplace = True)
            if ('language' not in data.columns and 'language of original document' in data.columns):
                data.rename(columns = {'language of original document': 'language'}, inplace = True)
            if ('note' not in data.columns and 'cited by' in data.columns):
                data.rename(columns = {'cited by': 'note'}, inplace = True)
            if ('page_count' not in data.columns and 'page count' in data.columns):
                data.rename(columns = {'page count': 'page_count'}, inplace = True)
            if ('pubmed_id' not in data.columns and 'pubmed id' in data.columns):
                data.rename(columns = {'pubmed id': 'pubmed_id'}, inplace = True)
            sanity_check = ['abbrev_source_title', 'abstract', 'address', 'affiliation', 'art_number',
                            'author', 'author_keywords', 'chemicals_cas', 'coden',      
                            'correspondence_address1', 'document_type', 'doi', 'editor', 'funding_details',
                            'funding_text\xa01', 'funding_text\xa02', 'funding_text\xa03', 'isbn', 'issn',
                            'journal', 'keywords', 'language', 'note', 'number', 'page_count', 'pages',
                            'publisher', 'pubmed_id', 'references', 'source', 'sponsors', 'title',
                            'tradenames', 'url', 'volume', 'year']
            for col in sanity_check:
                if (col not in data.columns):
                    data[col] = 'UNKNOWN'
            data           = data.reindex(sorted(data.columns), axis = 1)
            data['author'] = data['author'].apply(lambda x: x.replace(';', ' and ') if isinstance(x, str) else x)
            doc           = data.shape[0]
        else:
            f_file  = open(bib, 'r', encoding = 'utf8')
            f_lines = f_file.read()
            f_list  = f_lines.split('\n')
            if (db == 'wos'):
                f_list_ = []
                for i in range(0, len(f_list)):
                    if (f_list[i][:3] != '   '):
                        f_list_.append(f_list[i])
                    else:
                        if (f_list_[-1].find('Cited-References') != -1):
                            f_list[i] = f_list[i].replace(';', ',')
                        if (f_list_[-1].find('Cited-References') == -1):
                            f_list_[-1] = f_list_[-1] + f_list[i]
                        else:
                            f_list[i]   = f_list[i].replace(';', ',')
                            f_list_[-1] = f_list_[-1] + ';' + f_list[i]
                f_list = f_list_
            if (db == 'pubmed'):
                f_list_ = []
                for i in range(0, len(f_list)):
                    if (i == 0 and f_list[i][:6] != '      ' ):
                        f_list_.append(f_list[i])
                    elif (i > 0 and f_list[i][:6] != '      ' and f_list[i][:6] != f_list[i-1][:6] and (f_list[i][:6].lower() != 'fau - ' and f_list[i][:6].lower() != 'au  - ' and f_list[i][:6].lower() != 'auid- ' and f_list[i][:6].lower() != 'ad  - ')):
                        f_list_.append(f_list[i])
                    elif (i > 0 and f_list[i][:6] != '      ' and f_list[i][:6] == f_list[i-1][:6] and (f_list[i][:6].lower() != 'fau - ' and f_list[i][:6].lower() != 'au  - ' and f_list[i][:6].lower() != 'auid- ' and f_list[i][:6].lower() != 'ad  - ' and f_list[i][:6].lower() != 'pt  - ')):
                        f_list_[-1] = f_list_[-1] + '; ' + f_list[i][6:]
                    elif (f_list[i][:6] == '      '):
                        f_list_[-1] = f_list_[-1] + f_list[i][6:]
                    elif (f_list[i][:6] == 'FAU - '):
                        f_list_.append(f_list[i])
                        j = i + 1
                        while (len(f_list[j]) != 0):
                            j = j + 1
                            if (f_list[j][:6].lower() == 'fau - '):
                                f_list_[-1] = f_list_[-1] + '; ' + f_list[j][6:]
                                f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                    elif (f_list[i][:6] == 'AU  - '):
                        f_list_.append(f_list[i])
                        j = i + 1
                        while (len(f_list[j]) != 0):
                            j = j + 1
                            if (f_list[j][:6].lower() == 'au  - '):
                                f_list_[-1] = f_list_[-1] + ' and ' + f_list[j][6:]
                                f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                    elif (f_list[i][:6] == 'AUID- '):
                        f_list_.append(f_list[i])
                        j = i + 1
                        while (len(f_list[j]) != 0):
                            j = j + 1
                            if (f_list[j][:6].lower() == 'auid- '):
                                f_list_[-1] = f_list_[-1] + '; ' + f_list[j][6:]
                                f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                    elif (f_list[i][:6] == 'AD  - '):
                        f_list_.append(f_list[i])
                        j = i + 1
                        while (len(f_list[j]) != 0):
                            j = j + 1
                            if (f_list[j][:6].lower() == 'ad  - '):
                                f_list_[-1] = f_list_[-1] + f_list[j][6:]
                                f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                    elif (f_list[i][:6] == 'PT  - '):
                        f_list_.append(f_list[i])
                        j = i + 1
                        while (len(f_list[j]) != 0):
                            j = j + 1
                            if (f_list[j][:6].lower() == 'pt  - '):
                                f_list[j] = f_list[j][:6].lower() + f_list[j][6:]
                f_list = [item for item in f_list_]
                for i in range(0, len(f_list)):
                    if (len(f_list[i]) > 0):
                        if (f_list[i][4] == '-'):
                            f_list[i] = f_list[i][:4] + '=' + f_list[i][5:]
                        if (f_list[i][:3] == 'LID'):
                            f_list[i] = f_list[i].replace(' [doi]', '')
            lhs = []
            rhs = []
            doc = 0
            for i in range(0, len(f_list)):
              if (f_list[i].find('@') == 0 or f_list[i][:4].lower() == 'pmid'):  
                lhs.append('doc_start')
                rhs.append('doc_start')
                if (db == 'pubmed'):
                    lhs.append('note')
                    rhs.append('0')
                    lhs.append('source')
                    rhs.append('PubMed')
                if (db == 'wos'):
                    lhs.append('source')
                    rhs.append('WoS')
                doc = doc + 1
              if ( (f_list[i].find('=') != -1 and f_list[i].find(' ') != 0) or (f_list[i].find('=') != -1 and f_list[i].find('=') == 15) ): # DBLP
                lhs.append(f_list[i].split('=')[0].lower().strip())
                rhs.append(f_list[i].split('=')[1].replace('{', '').replace('},', '').replace('}', '').replace('}},', '').strip())
              elif (f_list[i].find(' ') == 0 and i!= 0 and rhs[-1] != 'doc_start'):
                rhs[-1] = rhs[-1]+' '+f_list[i].replace('{', '').replace('},', '').replace('}', '').replace('}},', '').strip()
            if (db == 'scopus' and 'abbrev_source_title' not in lhs and 'journal' in lhs):
                for i in range(0, len(lhs)):
                    if (lhs[i] == 'journal'):
                        lhs[i] = 'abbrev_source_title'
            if (db == 'scopus'):
                for i in range(0, len(lhs)):
                    if (lhs[i] == 'type'):
                        lhs[i] = 'document_type'
            if (db == 'wos' and 'journal-iso' not in lhs and 'journal' in lhs):
                for i in range(0, len(lhs)):
                    if (lhs[i] == 'journal'):
                        lhs[i] = 'journal-iso'
            if (db == 'pubmed' and 'ta' not in lhs and 'jt' in lhs):
                for i in range(0, len(lhs)):
                    if (lhs[i] == 'jt'):
                        lhs[i] = 'ta'
            if (db == 'wos'):
                for i in range(0, len(lhs)):
                    if (lhs[i] == 'affiliation'):
                        lhs[i] = 'affiliation_'
                    if (lhs[i] == 'affiliations'):
                        lhs[i] = 'affiliation'
                    if (lhs[i] == 'article-number'):
                        lhs[i] = 'art_number'
                    if (lhs[i] == 'cited-references'):
                        lhs[i] = 'references'
                    if (lhs[i] == 'keywords'):
                        lhs[i] = 'author_keywords'
                    if (lhs[i] == 'journal-iso'):
                        lhs[i] = 'abbrev_source_title'
                    if (lhs[i] == 'keywords-plus'):
                        lhs[i] = 'keywords'
                    if (lhs[i] == 'note'):
                        lhs[i] = 'note_'
                    if (lhs[i] == 'times-cited'):
                        lhs[i] = 'note'
                    if (lhs[i] == 'type'):
                        lhs[i] = 'document_type'
                    lhs[i] = lhs[i].replace('-', '_')
            if (db == 'pubmed'):
                for i in range(0, len(lhs)):
                    if (lhs[i] == 'ab'):
                        lhs[i] = 'abstract'
                    if (lhs[i] == 'ad'):
                        lhs[i] = 'affiliation'
                    if (lhs[i] == 'au'):
                        lhs[i] = 'author'
                    if (lhs[i] == 'auid'):
                        lhs[i] = 'orcid'
                    if (lhs[i] == 'fau'):
                        lhs[i] = 'full_author'
                    if (lhs[i] == 'lid'):
                        lhs[i] = 'doi'
                    if (lhs[i] == 'dp'):
                        lhs[i] = 'year'
                        rhs[i] = rhs[i][:4]
                    if (lhs[i] == 'ed'):
                        lhs[i] = 'editor'
                    if (lhs[i] == 'ip'):
                        lhs[i] = 'issue'
                    if (lhs[i] == 'is'):
                        lhs[i] = 'issn'
                    if (lhs[i] == 'isbn'):
                        lhs[i] = 'isbn'
                    if (lhs[i] == 'jt'):
                        lhs[i] = 'journal'
                    if (lhs[i] == 'la'):
                        lhs[i] = 'language'
                        if (rhs[i] in self.language_names.keys()):
                            rhs[i] = self.language_names[rhs[i]]
                    if (lhs[i] == 'mh'):
                        lhs[i] = 'keywords'
                    if (lhs[i] == 'ot'):
                        lhs[i] = 'author_keywords'
                    if (lhs[i] == 'pg'):
                        lhs[i] = 'pages'
                    if (lhs[i] == 'pt'):
                        lhs[i] = 'document_type'
                    if (lhs[i] == 'pmid'):
                        lhs[i] = 'pubmed_id'
                    if (lhs[i] == 'ta'):
                        lhs[i] = 'abbrev_source_title'
                    if (lhs[i] == 'ti'):
                        lhs[i] = 'title'
                    if (lhs[i] == 'vi'):
                        lhs[i] = 'volume'
            labels       = list(set(lhs))
            labels.remove('doc_start')
            sanity_check = ['abbrev_source_title', 'abstract', 'address', 'affiliation', 'art_number', 'author', 'author_keywords', 'chemicals_cas', 'coden', 'correspondence_address1', 'document_type', 'doi', 'editor', 'funding_details', 'funding_text\xa01', 'funding_text\xa02', 'funding_text\xa03', 'isbn', 'issn', 'journal', 'keywords', 'language', 'note', 'number', 'page_count', 'pages', 'publisher', 'pubmed_id', 'references', 'source', 'sponsors', 'title', 'tradenames', 'url', 'volume', 'year']
            for item in sanity_check:
                if (item not in labels):
                    labels.append(item)
            labels.sort()      
            values      = [i for i in range(0, len(labels))] 
            labels_dict = dict(zip(labels, values))
            data        = pd.DataFrame(index = range(0, doc), columns = labels)
            count       = -1
            for i in range(0, len(rhs)):
              if (lhs[i] == 'doc_start'):
                count = count + 1
              else:
                data.iloc[count, labels_dict[lhs[i]]] = rhs[i]
        
        entries = list(data.columns)
        
        # WoS -> Scopus
        data['document_type'] = data['document_type'].replace('Article; Early Access','Article in Press')
        data['document_type'] = data['document_type'].replace('Article; Proceedings Paper','Proceedings Paper')
        data['document_type'] = data['document_type'].replace('Article; Proceedings Paper','Proceedings Paper')
        data['document_type'] = data['document_type'].replace('Article; Discussion','Article')
        data['document_type'] = data['document_type'].replace('Article; Letter','Article')
        data['document_type'] = data['document_type'].replace('Article; Excerpt','Article')
        data['document_type'] = data['document_type'].replace('Article; Chronology','Article')
        data['document_type'] = data['document_type'].replace('Article; Correction','Article')
        data['document_type'] = data['document_type'].replace('Article; Correction, Addition','Article')
        data['document_type'] = data['document_type'].replace('Article; Data Paper','Article')
        data['document_type'] = data['document_type'].replace('Art Exhibit Review','Review')
        data['document_type'] = data['document_type'].replace('Dance Performance Review','Review')
        data['document_type'] = data['document_type'].replace('Music Performance Review','Review')
        data['document_type'] = data['document_type'].replace('Music Score Review','Review')
        data['document_type'] = data['document_type'].replace('Film Review','Review')
        data['document_type'] = data['document_type'].replace('TV Review, Radio Review','Review')
        data['document_type'] = data['document_type'].replace('TV Review, Radio Review, Video','Review')
        data['document_type'] = data['document_type'].replace('Theater Review, Video','Review')
        data['document_type'] = data['document_type'].replace('Database Review','Review')
        data['document_type'] = data['document_type'].replace('Record Review','Review')
        data['document_type'] = data['document_type'].replace('Software Review','Review')
        data['document_type'] = data['document_type'].replace('Hardware Review','Review')
        
        # PubMed -> Scopus
        data['document_type'] = data['document_type'].replace('Clinical Study','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial Protocol','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase I','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase II','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase III','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase IV','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Veterinary','Article')
        data['document_type'] = data['document_type'].replace('Comparative Study','Article')
        data['document_type'] = data['document_type'].replace('Controlled Clinical Trial','Article')
        data['document_type'] = data['document_type'].replace('Corrected and Republished Article','Article')
        data['document_type'] = data['document_type'].replace('Duplicate Publication','Article')
        data['document_type'] = data['document_type'].replace('Essay','Article')
        data['document_type'] = data['document_type'].replace('Historical Article','Article')
        data['document_type'] = data['document_type'].replace('Journal Article','Article')
        data['document_type'] = data['document_type'].replace('Letter','Article')
        data['document_type'] = data['document_type'].replace('Meta-Analysis','Article')
        data['document_type'] = data['document_type'].replace('Randomized Controlled Trial','Article')
        data['document_type'] = data['document_type'].replace('Randomized Controlled Trial, Veterinary','Article')
        data['document_type'] = data['document_type'].replace('Research Support, N.I.H., Extramural','Article')
        data['document_type'] = data['document_type'].replace('Research Support, N.I.H., Intramural','Article')
        data['document_type'] = data['document_type'].replace("Research Support, Non-U.S. Gov't",'Article')
        data['document_type'] = data['document_type'].replace("Research Support, U.S. Gov't, Non-P.H.S.",'Article')
        data['document_type'] = data['document_type'].replace("Research Support, U.S. Gov't, P.H.S.",'Article')
        data['document_type'] = data['document_type'].replace('Research Support, U.S. Government','Article')
        data['document_type'] = data['document_type'].replace('Research Support, American Recovery and Reinvestment Act','Article')
        data['document_type'] = data['document_type'].replace('Technical Report','Article')
        data['document_type'] = data['document_type'].replace('Twin Study','Article')
        data['document_type'] = data['document_type'].replace('Validation Study','Article')
        data['document_type'] = data['document_type'].replace('Clinical Conference','Conference Paper')
        data['document_type'] = data['document_type'].replace('Congress','Conference Paper')
        data['document_type'] = data['document_type'].replace('Consensus Development Conference','Conference Paper')
        data['document_type'] = data['document_type'].replace('Consensus Development Conference, NIH','Conference Paper')
        data['document_type'] = data['document_type'].replace('Systematic Review','Review')
        data['document_type'] = data['document_type'].replace('Scientific Integrity Review','Review')
        
        if (del_duplicated == True and 'doi' in entries):
            duplicated = data['doi'].duplicated()
            title      = data['title']
            title      = title.to_list()
            title      = self.clear_text(title, stop_words  = [], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [])
            t_dupl     = pd.Series(title).duplicated()
            for i in range(0, duplicated.shape[0]):
                if (data.loc[i, 'doi'] == 'UNKNOWN' or pd.isnull(data.loc[i, 'doi'])):
                    duplicated[i] = False
                if (t_dupl[i] == True):
                    duplicated[i] = True
            idx        = list(duplicated.index[duplicated])
            data.drop(idx, axis = 0, inplace = True)
            data       = data.reset_index(drop = True)
            string_vb  = 'A Total of ' + str(doc-len(idx)) + ' Documents were Found ( ' + str(doc) + ' Documents and '+ str(len(idx)) + ' Duplicates )'
            self.vb.append(string_vb)
        else:
            string_vb  = 'A Total of ' + str(doc) + ' Documents were Found' 
            self.vb.append(string_vb)
        if (db == 'wos' and 'type' in entries):
            data['document_type'] = data['type']
        if ('document_type' in entries):
            types     = list(data['document_type'].replace(np.nan, 'UNKNOWN'))
            u_types   = list(set(types))
            u_types.sort()
            string_vb = ''
            self.vb.append(string_vb)
            for tp in u_types:
                string_vb = tp + ' = ' + str(types.count(tp))
                self.vb.append(string_vb)
        data.fillna('UNKNOWN', inplace = True)
        data['keywords']        = data['keywords'].apply(lambda x: x.replace(',',';'))
        data['author_keywords'] = data['author_keywords'].apply(lambda x: x.replace(',',';'))
        if db == 'wos':
            if 'affiliation_' not in data.columns:
                data['affiliation_'] = 'UNKNOWN'
        if (db == 'wos'):
            idx     = data[data['year'] == 'UNKNOWN'].index.values
            idx_val = [data.loc[i, 'da'][:4] for i in idx]
            for i in range(0, len(idx)):
                data.iloc[idx[i], -1] = idx_val[i]
        if ('affiliation' in data.columns and 'affiliation_' in data.columns):
            filtered_indices = data[(data['affiliation'] == 'UNKNOWN') & (data['affiliation_'] != 'UNKNOWN')].index
            filtered_indices = list(filtered_indices)
            for i in filtered_indices:
                s         = data.loc[i, 'affiliation_']
                parts     = s.split('.')
                parts[0]  = re.sub(r'.*?\(Corresponding Author\), ', '', parts[0]) 
                new_parts = [part.split(',', 1)[-1].strip() for part in parts[1:] if ',' in part]
                new_parts.insert(0, parts[0])
                new_s     = '. '.join(new_parts)
                data.loc[i, 'affiliation'] =  new_s
        if ('affiliation' in data.columns and 'affiliations' in data.columns):
            filtered_indices = data[(data['affiliation'] == 'UNKNOWN') & (data['affiliations'] != 'UNKNOWN')].index
            filtered_indices = list(filtered_indices)
            for i in filtered_indices:
                data.loc[i, 'affiliation'] =  data.loc[i, 'affiliations']
        if (db == 'scopus'):
            data['correspondence_address1'] = ('Corresponding Author ' + data['correspondence_address1'] )
            data['affiliation']             = data.apply(map_authors_to_affiliations, axis = 1)
        if (db == 'pubmed'):
            data['affiliation'] = data.apply(lambda row: assign_authors_to_affiliations(row['author'], row['affiliation']), axis = 1)
        if (db == 'wos'):
            data['affiliation_'] = data['affiliation_'].str.replace(r'(?<=[A-Z])\.', '#' , regex = True)
            data['affiliation_'] = data['affiliation_'].str.replace(';', ',', regex = False).str.replace('.', ';', regex = False).str.rstrip(';')
            data['affiliation_'] = data['affiliation_'].str.replace('#', '.', regex = False)
        data['abstract'] = data['abstract'].str.replace('[No abstract available]', 'UNKNOWN', regex = False)
        data             = data.reindex(sorted(data.columns), axis = 1)
        if (db == 'scopus'):
            data["abbrev_source_title"] = np.where( (data["abbrev_source_title"] == "UNKNOWN") & (data["source title"] != "UNKNOWN"), data["source title"], data["abbrev_source_title"] )
        return data, entries
    
    # Function: Update Verbose
    def __update_vb(self):
        self.vb   = []
        self.vb.append('A Total of ' + str(self.data.shape[0]) + ' Documents Remains' )
        types     = list(self.data['document_type'])
        u_types   = list(set(types))
        u_types.sort()
        string_vb = ''
        self.vb.append(string_vb)
        for tp in u_types:
            string_vb = tp + ' = ' + str(types.count(tp))
            self.vb.append(string_vb)
        return
    
    ##############################################################################
    
    # Function: Get Entries
    def __get_str(self, entry = 'references', s = ';', lower = True, sorting = True):
        #----------------------------------------------------------------------
               
        def is_year_parentheses(ref):
            cleaned = re.sub(r'^[\s,;:.()\[\]{}]+|[\s,;:.()\[\]{}]+$', '', ref)
            return bool(re.fullmatch(r'\d{4}', cleaned))
        
        #----------------------------------------------------------------------
        
        column      = self.data[entry]
        if entry != 'references':
            info = [ [ ' '.join(item.split()).lower() if lower else ' '.join(item.split()) for item in e.split(s) if item.strip() and item.strip() != 'note' ] if isinstance(e, str) else [] for e in column ]
        else:
            info = [ [ (' '.join(item.split()).lower() if lower else ' '.join(item.split())) for item in e.split(s) if item.strip() and item.strip() != 'note' and not is_year_parentheses(item) ] if isinstance(e, str) else [] for e in column ]
        unique_info = list({item for sublist in info for item in sublist})
        if ('' in unique_info):
            unique_info.remove('')
        if (sorting == True):
            unique_info.sort()
        return info, unique_info
   
    # Function: Get Citations
    def __get_citations(self, series):
        
        #----------------------------------------------------------------------
        
        def extract_numeric(value):
            try:
                idx = value.find(';')
                if idx >= 0:
                    return int(value[:idx])
                return int(value)
            except ValueError:
                match = re.search(r'\d+', value)
                return int(match.group()) if match else 0
            
        #----------------------------------------------------------------------
        
        series = series.fillna('unknown').str.lower()
        series = series.str.replace('cited by:?', '', regex = True)
        return series.apply(extract_numeric).tolist()

    # Function: Get Past Citations per Year
    def __get_past_citations_year(self):
        df             = self.data[['author', 'title', 'doi', 'year', 'references']].sort_values(by = 'year').reset_index(drop = True)
        c_count        = [0] * df.shape[0]
        c_year         = df['year'].tolist()
        ref_lower      = [[ref.lower() for ref in refs] for refs in self.ref]
        title_to_index = {df.iloc[i, 1].lower(): i for i in range(df.shape[0])}
        for j, refs in enumerate(ref_lower):
            for ref in refs:
                for title, idx in title_to_index.items():
                    if (title in ref):
                        c_count[j] = c_count[j] + 1
        year_to_count = {year: 0 for year in sorted(set(c_year))}
        for year, count in zip(c_year, c_count):
            year_to_count[year] = year_to_count[year] + count
        c_year_  = list(year_to_count.keys())
        c_count_ = list(year_to_count.values())
        return c_year_, c_count_
    
    # Function: Get Countries
    def __get_countries(self):
                
        #----------------------------------------------------------------------
        
        def preprocess_affiliation(row):
            source = row['source'].lower()
            if (source in ['scopus', 'pubmed']):
                return row['affiliation']
            elif (source == 'wos'):
                aff = row['affiliation_'].replace('(Corresponding Author)', '')
                return aff.replace(',', ', ') if ',' in aff and ', ' not in aff else aff
            return 'UNKNOWN'
        
        def get_additional_country_data():
            unique_countries = set()
            for country_data in self.author_country_map.values():
                for _, country in country_data:
                    unique_countries.add(country)
            u_ctr = list(unique_countries)
            ctr   = []
            for index, row in self.data.iterrows():
                row_countries = []
                for author in self.aut[index]: 
                    if (author in self.author_country_map):
                        for row_idx, country in self.author_country_map[author]:
                            if (row_idx == index):
                                row_countries.append(country)
                ctr.append(row_countries)
            self.corr_a_country_map = {}
            for index, row in self.data.iterrows():
                if (self.database.lower() == 'wos'):
                    if ('Corresponding Author' in row['affiliation_']):
                        corresponding_author = next( (author for author in self.aut[index] if 'Corresponding Author' in row['affiliation_']), None )
                        if (corresponding_author and corresponding_author in self.author_country_map):
                            self.corr_a_country_map[corresponding_author] = self.author_country_map[corresponding_author]
                else:
                    corresponding_author = next((author for author in self.aut[index]  if 'corresponding author' in row['correspondence_address1'].lower()), None)
                    if (corresponding_author and corresponding_author in self.author_country_map):
                        self.corr_a_country_map[corresponding_author] = self.author_country_map[corresponding_author]
            self.frst_a_country_map = {}
            for index, row in self.data.iterrows():
                if (self.aut[index]):  
                    first_author = self.aut[index][0]
                    if (first_author in self.author_country_map):
                        self.frst_a_country_map[first_author] = self.author_country_map[first_author]
            return ctr, u_ctr
        
        #----------------------------------------------------------------------
        
        data = self.data.copy(deep    = True)
        data['processed_affiliation'] = data.apply(preprocess_affiliation, axis = 1).str.lower()
        country_replacements          = {
                                            ' usa':            ' united states of america',
                                            'england':         'united kingdom',
                                            'antigua & barbu': 'antigua and barbuda',
                                            'bosnia & herceg': 'bosnia and herzegovina',
                                            'cent afr republ': 'central african republic',
                                            'czech republic':  'czechia',
                                            'dominican rep':   'dominican republic',
                                            'equat guinea':    'equatorial guinea',
                                            'fr austr lands':  'french southern territories',
                                            'fr polynesia':    'french polynesia',
                                            'malagasy republ': 'madagascar',
                                            'mongol peo rep':  'mongolia',
                                            'neth antilles':   'saint martin',
                                            'north ireland':   'ireland',
                                            'peoples r china': 'china',
                                            'rep of georgia':  'georgia',
                                            'russia':          'russian federation',
                                            'sao tome e prin': 'sao tome and principe',
                                            'scotland':        'united kingdom',
                                            'st kitts & nevi': 'saint kitts and nevis',
                                            'trinid & tobago': 'trinidad and tobago',
                                            'u arab emirates': 'united arab emirates',
                                            'usa':             'united states of america',
                                            'vietnam':         'viet nam',
                                        }
        data['processed_affiliation'] = data['processed_affiliation'].replace('united states of america', 'usa', regex = True)
        data['processed_affiliation'] = data['processed_affiliation'].replace('united states', 'usa', regex = True)
        data['processed_affiliation'] = data['processed_affiliation'].replace(country_replacements, regex = True)
        self.author_country_map       = {author: [] for author in self.u_aut}
        for index, row in data.iterrows():
            affiliations = row['processed_affiliation'].split(';')
            authors      = self.aut[row.name]
            for author in authors:
                detected_country = 'UNKNOWN'
                for aff in affiliations:
                    for country in self.country_names:
                        if (country.lower() in aff and author in aff):
                            detected_country = country
                            break
                    if (detected_country != 'UNKNOWN'):
                        break
                if not any(entry[0] == index for entry in self.author_country_map[author]):
                    self.author_country_map[author].append((index, detected_country))
        ctr, u_ctr = get_additional_country_data()
        return ctr, u_ctr 

    # Function: Replace Unknows  
    def replace_unknowns(self, list_of_lists):
        new_list = []
        for sublist in list_of_lists:
            new_sublist = []
            prev_value  = None
            for item in sublist:
                if (item == 'UNKNOWN'):
                    new_sublist.append(prev_value if prev_value is not None else 'UNKNOWN')
                else:
                    new_sublist.append(item)
                    prev_value = item
            new_list.append(new_sublist)
        return new_list
    
    # Function: Get Institutions   
    def __get_institutions(self):
        
        #----------------------------------------------------------------------

        def extract_top_institution_with_priority(text, institution_names):
            segments              = text.split(';')
            selected_institutions = []
            for segment in segments:
                segment   = segment.strip().lower()
                sub_parts = segment.split(',')
                matches   = [ (part.strip(), self.inst_priority.get(keyword, 0)) for part in sub_parts for keyword in self.inst_priority.keys() if keyword in part ]
                if (matches):
                    matches.sort(key = lambda x: (-x[1], -len(x[0])))
                    selected_institutions.append(matches[0][0])
                else:
                    selected_institutions.append('UNKNOWN')
            return selected_institutions

        #----------------------------------------------------------------------

        sources                = self.data['source'].str.lower()
        affiliations           = (self.data['affiliation'].fillna('').str.lower() if 'affiliation' in self.data.columns else pd.Series([''] * len(self.data)) )
        affiliations_wos       = (self.data['affiliation_'].fillna('').str.lower() if 'affiliation_' in self.data.columns else pd.Series([''] * len(self.data)) )
        affiliations_wos       = (self.data['affiliation_'].fillna('').str.lower() if 'affiliation_' in self.data.columns else pd.Series([''] * len(self.data)) )
        processed_affiliations = np.where(sources.isin(['scopus', 'pubmed']), affiliations, np.where(sources == 'wos', affiliations_wos, 'UNKNOWN') )
        processed_affiliations = pd.Series(processed_affiliations)
        top_institutions       = processed_affiliations.apply( lambda row: extract_top_institution_with_priority(row, self.institution_names))
        inst                   = [top for top in top_institutions]
        #flattened_institutions = [institution for sublist in top_institutions for institution in sublist]
        #u_inst                 = list(set(flattened_institutions))
        #u_inst                 = [re.sub(r'^(?:[A-Za-z]\.\s?)+', '', name) for name in u_inst]
        #u_inst                 = list(set(u_inst))
        self.author_inst_map   = {author: [] for author in self.u_aut}
        for index, institutions in enumerate(inst):
            for author in self.aut[index]:
                for institution in inst[index]:
                    for _, aff in enumerate(processed_affiliations[index].split(';')):
                        if (author in aff and institution in aff):
                            self.author_inst_map[author].append((index, re.sub(r'^(?:[A-Za-z]\.\s?)+', '', institution)))
                if (len(self.author_inst_map[author]) == 0):
                    self.author_inst_map[author].append((index, 'UNKNOWN'))
        self.author_inst_map = {k: list(set(v)) for k, v in self.author_inst_map.items()}
        #inst                 = []
        #for index, row in self.data.iterrows():
            #row_inst = []
            #for author in self.aut[index]: 
                #if (author in self.author_inst_map):
                    #for row_idx, uni in self.author_inst_map[author]:
                        #if (row_idx == index):
                            #row_inst.append(re.sub(r'^(?:[A-Za-z]\.\s?)+', '', uni))
            #inst.append(row_inst)
        inst                 = []
        for index, authors in enumerate(self.aut):
            row_inst = []
            for author in authors:
                aff_list = [uni for row_idx, uni in self.author_inst_map.get(author, []) if row_idx == index]
                aff      = aff_list[0] if aff_list else 'UNKNOWN'
                aff      = re.sub(r'^(?:[A-Za-z]\.\s?)+', '', aff)
                row_inst.append(aff)
            inst.append(row_inst)
        u_inst               = list({re.sub(r'^(?:[A-Za-z]\.\s?)+', '', name).strip() for row in inst for name in row })
        self.corr_a_inst_map = {}
        for index, row in self.data.iterrows():
            if (self.database == 'wos'):
                if ('corresponding author' in row['affiliation_'].lower()):
                    corresponding_author = next((author for author in self.aut[index] if 'corresponding author' in row['affiliation_'].lower()), None)
                    if (corresponding_author and corresponding_author in self.author_inst_map):
                        self.corr_a_inst_map[corresponding_author] = self.author_inst_map[corresponding_author]
            else:
                if ('corresponding author' in row['correspondence_address1'].lower()):
                    corresponding_author = next( (author for author in self.aut[index] if 'corresponding author' in row['correspondence_address1'].lower()), None)
                    if (corresponding_author and corresponding_author in self.author_inst_map):
                        self.corr_a_inst_map[corresponding_author] = self.author_inst_map[corresponding_author]
        self.frst_a_inst_map = {}
        for index, row in self.data.iterrows():
            if (self.aut[index]):  
                first_author = self.aut[index][0]
                if (first_author in self.author_inst_map):
                    self.frst_a_inst_map[first_author] = self.author_inst_map[first_author]
        return inst, u_inst

    # Function: Get Counts
    def __get_counts(self, u_ent, ent, acc = []):
        counts = []
        for u in u_ent:
            ents = 0
            for j, e in enumerate(ent):
                if (u in e):
                    if (acc):
                        ents = ents + acc[j]
                    else:
                        ents = ents + 1
            counts.append(ents)
        return counts

    # Function: Get Count Year
    def __get_counts_year(self, u_ent, ent):
        years = list(range(self.date_str, self.date_end+1))
        df_counts = pd.DataFrame(np.zeros((len(u_ent),len(years))))
        for i in range(0, len(u_ent)):
            for j in range(0, len(ent)):
                if (u_ent[i] in ent[j]):
                    k = years.index(int(self.dy[j]))
                    df_counts.iloc[i, k] = df_counts.iloc[i, k] + 1
        return df_counts
    
    # Function: Get Collaboration Year
    def __get_collaboration_year(self):
        max_aut         = list(set([str(item) for item in self.aut_docs]))
        max_aut         = sorted(max_aut, key = self.natsort)
        n_collaborators = ['n = ' + i for i in max_aut]
        n_collaborators.append('ci')
        years           = list(range(self.date_str, self.date_end+1))
        years           = [str(int(item)) for item in years]
        years.append('Total')
        dy_collab_year = pd.DataFrame(np.zeros((len(years), len( n_collaborators))), index = years, columns = n_collaborators)
        for k in range(0, len(self.aut)):
            i                        = str(int(self.dy[k]))
            j                        = ['n = ' + str(len(self.aut[k]))]
            dy_collab_year.loc[i, j] = dy_collab_year.loc[i, j] + 1    
        dy_collab_year.iloc[-1, :] = dy_collab_year.sum(axis = 0)
        dy_collab_year.iloc[:, -1] = dy_collab_year.sum(axis = 1)
        for i in range(0, dy_collab_year.shape[0]):
            ci                         = sum([ (j+1) * dy_collab_year.iloc[i, j] for j in range(0, dy_collab_year.shape[1]-1)])
            if (dy_collab_year.iloc[i, -1] > 0):
                dy_collab_year.iloc[i, -1] = round(ci / dy_collab_year.iloc[i, -1], 2)               
        return dy_collab_year
    
    # Function: Get Reference Year
    def __get_ref_year(self):
        date_end        = self.date_end
        year_pattern    = re.compile(r'(?<!\d)(\d{4})(?!\d)')
        extracted_years = []
        for ref in self.u_ref:
            matches     = year_pattern.findall(ref)
            valid_years = [int(year) for year in matches if 1665 <= int(year) <= date_end] # The oldest scientific journal is Philosophical Transactions, which was launched in 1665 by Henry Oldenburg
            extracted_years.append(max(valid_years) if valid_years else -1)
        return extracted_years

    # Function: Get Reference ID
    def __get_ref_id(self):
        labels_r       = ['r_' + str(i) for i in range(0, len(self.u_ref))]
        sources        = self.data['source'].str.lower()
        keys_1         = self.data['title'].str.lower().str.replace('[', '', regex = False).str.replace(']', '', regex = False).tolist()
        keys_2         = self.data['doi'].str.lower().tolist()
        keys           = np.where(sources.isin(['scopus', 'pubmed']), keys_1, np.where(sources == 'wos', keys_2, None))
        corpus          = ' '.join(ref.lower() for ref in self.u_ref)
        matched_indices = []
        for i, key in enumerate(keys):
            if (key and key.strip()):
                try:
                    compiled_regex = re.compile(key)
                    if (re.search(compiled_regex, corpus)):
                        matched_indices.append(i)
                except:
                    pass
        insd_r      = []
        insd_t      = []
        u_ref_lower = [ref.lower() for ref in self.u_ref]
        for i in matched_indices:
            key = keys[i]
            for j, ref in enumerate(u_ref_lower):
                if (re.search(key, ref)):
                    insd_r.append(f'r_{j}')
                    insd_t.append(str(i))
                    self.dy_ref[j] = int(self.dy[i])
                    break
        dict_lbs = dict(zip(insd_r, insd_t))
        dict_lbs.update({label: label for label in labels_r if label not in dict_lbs})
        labels_r = [dict_lbs.get(label, label) for label in labels_r]
        return labels_r
    
    ##############################################################################
    
    # Function: Author Profile
    def profiling_author(self, label_name = None, label_id = None, topn = 5):
        if label_id == None:
            name  = label_name.lower()
        else:
            name  = self.u_aut[int(label_id.split('_')[1])]
        paper_idx = self.author_to_papers.get(name, [])
        idx       = self.u_aut.index(name)
        citation  = [self.citation[i] for i in paper_idx]
        uni       = []
        ctr       = []
        for i in paper_idx:
            for inst, author in zip(self.uni[i], self.aut[i]):
                if inst.strip().upper() != 'UNKNOWN' and author == name:
                    uni.append(inst.strip())
            for country, author in zip(self.ctr[i], self.aut[i]):
                if country.strip().upper() != 'UNKNOWN' and author == name:
                    ctr.append(country.strip())
        profile   =  pd.DataFrame({
                    'Entity':             [name],
                    'Affiliation':        [list(set(uni))],
                    'Country':            [list(set(ctr))],
                    'Total Publications': [len(paper_idx)],
                    'Total Citations':    [np.sum(citation) if citation else 0],
                    'Avg Citations':      [round(np.mean(citation), 2) if citation else 0],
                    'h-index':            [self.aut_h[idx]],
                    'g-index':            [self.aut_g[idx]],
                    'e-index':            [round(self.aut_e[idx], 2)],
                    'First Year':          [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().min()) else None],
                    'Last Year':           [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().max()) else None],
                    'Top Journals':       [Counter(item.strip().lower() for sublist in [self.jou[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords':       [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords+':      [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Co-Entities':    [Counter(item.strip().lower() for sublist in [self.aut[i] for i in paper_idx] for item in sublist if item != name).most_common(topn)],
                    'Papers ID':          [paper_idx],
                    'Papers Citation':    [citation],
                    'Papers Year':        [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile
    
    # Function: Affiliation Profile
    def profiling_affiliation(self, label_name = None, label_id = None, topn = 5):
        if label_id == None:
            name  = label_name.lower()
        else:
            name  = self.u_uni[int(label_id.split('_')[1])]
        paper_idx = [i for i, sublist in enumerate(self.uni) if any(name == inst.lower().strip() for inst in sublist)]
        citation  = [self.citation[i] for i in paper_idx]
        aut       = []
        ctr       = []
        for i in paper_idx:
            for author, inst in zip(self.aut[i], self.uni[i]):
                if author.strip().upper() != 'UNKNOWN' and inst == name:
                    aut.append(author.strip())
            for country, inst in zip(self.ctr[i], self.uni[i]):
                if country.strip().upper() != 'UNKNOWN' and inst == name:
                    ctr.append(country.strip())    
        profile   =  pd.DataFrame({
                    'Entity':              [name],
                    'Authors':             [list(set(aut))],
                    'Country':             [list(set(ctr))],
                    'Total Publications':  [len(paper_idx)],
                    'Total Citations':     [np.sum(citation) if citation else 0],
                    'Avg Citations':       [round(np.mean(citation), 2) if citation else 0],
                    'First Year':          [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().min()) else None],
                    'Last Year':           [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().max()) else None],
                    'Top Journals':        [Counter(item.strip().lower() for sublist in [self.jou[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords':        [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords+':       [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Co-Entities':     [Counter(item.strip().lower() for sublist in [self.uni[i] for i in paper_idx] for item in sublist if item != name).most_common(topn)],
                    'Papers ID':           [paper_idx],
                    'Papers Citation':     [citation],
                    'Papers Year':         [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile
    
    # Function: Country Profile
    def profiling_country(self, label_name = None, label_id = None, topn = 5):
        if label_id == None:
            name  = label_name.lower()
        else:
            name  = self.u_ctr[int(label_id.split('_')[1])]
        paper_idx = [i for i, sublist in enumerate(self.ctr) if any(name == ctr.strip() for ctr in sublist)]
        citation  = [self.citation[i] for i in paper_idx]
        aut       = []
        uni       = []
        for i in paper_idx:
            for author, country in zip(self.aut[i], self.ctr[i]):
                if  author.strip().upper() != 'UNKNOWN' and country == name:
                    aut.append(author.strip())
            for inst, country in zip(self.uni[i], self.ctr[i]):
                if inst.strip().upper() != 'UNKNOWN' and country == name:
                    uni.append(inst.strip())       
        profile   =  pd.DataFrame({
                    'Entity':              [name],
                    'Authors':             [list(set(aut))],
                    'Affiliation':         [list(set(uni))],
                    'Total Publications':  [len(paper_idx)],
                    'Total Citations':     [np.sum(citation) if citation else 0],
                    'Avg Citations':       [round(np.mean(citation), 2) if citation else 0],
                    'First Year':          [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().min()) else None],
                    'Last Year':           [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().max()) else None],
                    'Top Journals':        [Counter(item.strip().lower() for sublist in [self.jou[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords':        [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords+':       [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Co-Entities':     [Counter(item.strip().lower() for sublist in [self.ctr[i] for i in paper_idx] for item in sublist if item != name).most_common(topn)],
                    'Papers ID':           [paper_idx],
                    'Papers Citation':     [citation],
                    'Papers Year':         [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile
    
    # Function: Journal Profile
    def profiling_journal(self, label_name = None, label_id = None, topn = 5):
        if label_id == None:
            name  = label_name.lower()
        else:
            name  = self.u_jou[int(label_id.split('_')[1])]
        paper_idx = [i for i, sublist in enumerate(self.jou) if any(name == jou.lower().strip() for jou in sublist)]
        citation  = [self.citation[i] for i in paper_idx]
        aut       = []
        uni       = []
        ctr       = []
        for i in paper_idx:
            for author, journal in zip(self.aut[i], self.jou[i]):
                if author.strip().upper() != 'UNKNOWN' and journal == name:
                    aut.append(author.strip())
            for inst, journal in zip(self.uni[i], self.jou[i]):
                if inst.strip().upper() != 'UNKNOWN' and journal == name:
                    uni.append(inst.strip())  
            for country, journal in zip(self.ctr[i], self.jou[i]):
                if country.strip().upper() != 'UNKNOWN' and journal == name:
                    ctr.append(country.strip())
        profile   =  pd.DataFrame({
                    'Entity':              [name],
                    'Authors':             [list(set(aut))],
                    'Affiliation':         [list(set(uni))],
                    'Country':             [list(set(ctr))],
                    'Total Publications':  [len(paper_idx)],
                    'Total Citations':     [np.sum(citation) if citation else 0],
                    'Avg Citations':       [round(np.mean(citation), 2) if citation else 0],
                    'First Year':          [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().min()) else None],
                    'Last Year':           [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().max()) else None],
                    'Top Keywords':        [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords+':       [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Papers ID':           [paper_idx],
                    'Papers Citation':     [citation],
                    'Papers Year':         [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile
      
    # Function: Keyword Profile
    def profiling_keyword(self, label_name = None, label_id = None, topn = 5):
        if label_id == None:
            name  = label_name.lower()
        else:
            name  = self.u_auk[int(label_id.split('_')[1])]
        paper_idx = [i for i, sublist in enumerate(self.auk) if any(name == kwa.lower().strip() for kwa in sublist)]
        citation  = [self.citation[i] for i in paper_idx]
        aut       = []
        uni       = []
        ctr       = []
        for i in paper_idx:
            for author, keyword in zip(self.aut[i], self.auk[i]):
                if  author.strip().upper() != 'UNKNOWN' and keyword == name:
                    aut.append(author.strip())
            for inst, keyword in zip(self.uni[i], self.auk[i]):
                if inst.strip().upper() != 'UNKNOWN' and keyword == name:
                    uni.append(inst.strip())  
            for country, keyword in zip(self.ctr[i], self.auk[i]):
                if country.strip().upper() != 'UNKNOWN' and keyword == name:
                    ctr.append(country.strip())
        profile   =  pd.DataFrame({
                    'Entity':              [name],
                    'Authors':             [list(set(aut))],
                    'Affiliation':         [list(set(uni))],
                    'Country':             [list(set(ctr))],
                    'Total Publications':  [len(paper_idx)],
                    'Total Citations':     [np.sum(citation) if citation else 0],
                    'Avg Citations':       [round(np.mean(citation), 2) if citation else 0],
                    'First Year':          [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().min()) else None],
                    'Last Year':           [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().max()) else None],
                    'Top Journals':        [Counter(item.strip().lower() for sublist in [self.jou[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords+':       [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Co-Entities':     [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist if item != name).most_common(topn)],
                    'Papers ID':           [paper_idx],
                    'Papers Citation':     [citation],
                    'Papers Year':         [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile
    
    # Function: Keyword Plus Profile
    def profiling_keyword_plus(self, label_name = None, label_id = None, topn = 5):
        if label_id == None:
            name  = label_name.lower()
        else:
            name  = self.u_kid[int(label_id.split('_')[1])]
        paper_idx = [i for i, sublist in enumerate(self.kid) if any(name == kwp.lower().strip() for kwp in sublist)]
        citation  = [self.citation[i] for i in paper_idx]
        aut       = []
        uni       = []
        ctr       = []
        for i in paper_idx:
            for author, keyword in zip(self.aut[i], self.kid[i]):
                if  author.strip().upper() != 'UNKNOWN' and keyword == name:
                    aut.append(author.strip())
            for inst, keyword in zip(self.uni[i], self.kid[i]):
                if inst.strip().upper() != 'UNKNOWN' and keyword == name:
                    uni.append(inst.strip())  
            for country, keyword in zip(self.ctr[i], self.kid[i]):
                if country.strip().upper() != 'UNKNOWN' and keyword == name:
                    ctr.append(country.strip())
        profile   =  pd.DataFrame({
                    'Entity':              [name],
                    'Authors':             [list(set(aut))],
                    'Affiliation':         [list(set(uni))],
                    'Country':             [list(set(ctr))],
                    'Total Publications':  [len(paper_idx)],
                    'Total Citations':     [np.sum(citation) if citation else 0],
                    'Avg Citations':       [round(np.mean(citation), 2) if citation else 0],
                    'First Year':          [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().min()) else None],
                    'Last Year':           [int(y) if pd.notnull(y := self.data.iloc[paper_idx]['year'].dropna().max()) else None],
                    'Top Journals':        [Counter(item.strip().lower() for sublist in [self.jou[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords':        [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Co-Entities':     [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist if item != name).most_common(topn)],
                    'Papers ID':           [paper_idx],
                    'Papers Citation':     [citation],
                    'Papers Year':         [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile
    
    # Function: Reference Profile
    def profiling_reference(self, label_id = None, topn = 5):
        name      = self.u_ref[int(label_id.split('_')[1])]
        paper_idx = [i for i, sublist in enumerate(self.ref) if any(name == ref.strip() for ref in sublist)]
        citation  = [self.citation[i] for i in paper_idx]
        aut       = []
        uni       = []
        ctr       = []
        for i in paper_idx:
            for author in self.aut[i]:
                if author.strip().upper()  != 'UNKNOWN' and name in self.ref[i]:
                    aut.append(author.strip())
            for inst in self.uni[i]:
                if inst.strip().upper()    != 'UNKNOWN' and name in self.ref[i]:
                    uni.append(inst.strip())  
            for country in self.ctr[i]:
                if country.strip().upper() != 'UNKNOWN' and name in self.ref[i]:
                    ctr.append(country.strip())
        profile   =  pd.DataFrame({
                    'Entity':              [name],
                    'Authors':             [list(set(aut))],
                    'Affiliation':         [list(set(uni))],
                    'Country':             [list(set(ctr))],
                    'Total Citations':     [len(paper_idx)],
                    'Top Journals':        [Counter(item.strip().lower() for sublist in [self.jou[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords':        [Counter(item.strip().lower() for sublist in [self.auk[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Keywords+':       [Counter(item.strip().lower() for sublist in [self.kid[i] for i in paper_idx] for item in sublist).most_common(topn)],
                    'Top Co-Entities':     [Counter(item.strip().lower() for sublist in [self.ref[i] for i in paper_idx] for item in sublist if item != name).most_common(topn)],
                    'Papers ID':           [paper_idx],
                    'Papers Citation':     [citation],
                    'Papers Year':         [self.data.iloc[paper_idx]['year'].to_list()],
                                   }).T
        return profile

    ##############################################################################
    
    # Function: Wordcloud 
    def word_cloud_plot(self, entry = 'kwp', size_x = 10, size_y = 10, wordsn = 500, rmv_custom_words = []):   
        if entry == 'kwp':
            kid_    = [item for sublist in self.kid  for item in sublist]
            corpora = ' '.join(kid_)
            corpora = corpora.lower()
        elif entry == 'kwa':
            auk_    = [item for sublist in self.kid  for item in sublist]
            corpora = ' '.join(auk_)
            corpora = corpora.lower()
        elif entry == 'abs':
            abs_    = self.data['abstract']
            abs_    = list(abs_)
            abs_    = [x for x in abs_ if str(x) != 'nan']
            corpora = ' '.join(abs_)
            corpora = corpora.lower()
        elif entry == 'title':
            tit_    = self.data['title']
            tit_    = list(tit_)
            tit_    = [x for x in tit_ if str(x) != 'nan']
            corpora = ' '.join(tit_)
            corpora = corpora.lower()
        if len(rmv_custom_words) > 0:
            text    = corpora.split()
            text    = [x.replace(' ', '') for x in text if x.replace(' ', '') not in rmv_custom_words]
            corpora = ' '.join(text)

        wordcloud = WordCloud(
                                background_color = 'white',
                                max_words        = wordsn,
                                contour_width    = 25,
                                contour_color    = 'steelblue',
                                collocations     = False,
                                width            = 1600,
                                height           = 800
                             )
        wordcloud.generate(corpora)
        self.ask_gpt_wd = wordcloud.words_
        plt.figure(figsize = (size_x, size_y), facecolor = 'k')
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.show()
        return

    # Function: Get Top N-Grams 
    def get_top_ngrams(self, view = 'browser', entry = 'kwp', ngrams = 1, stop_words = [], rmv_custom_words = [], wordsn = 15):
        sw_full = []
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if  (entry == 'kwp'):
            corpora = pd.Series([' '.join(k) for k in self.kid]) 
        elif (entry == 'kwa'): 
            corpora = pd.Series([' '.join(a) for a in self.auk] )
        elif (entry == 'abs'):
            corpora = self.data['abstract']
        elif (entry == 'title'):
            corpora = self.data['title']
        if (len(stop_words) > 0):
            for sw_ in stop_words: 
                if   (sw_ == 'ar' or sw_ == 'ara' or sw_ == 'arabic'):
                    name = 'Stopwords-Arabic.txt'
                elif (sw_ == 'bn' or sw_ == 'ben' or sw_ == 'bengali'):
                    name = 'Stopwords-Bengali.txt'
                elif (sw_ == 'bg' or sw_ == 'bul' or sw_ == 'bulgarian'):
                    name = 'Stopwords-Bulgarian.txt'
                elif (sw_ == 'zh' or sw_ == 'chi' or sw_ == 'chinese'):
                    name = 'Stopwords-Chinese.txt'
                elif (sw_ == 'cs' or sw_ == 'cze' or sw_ == 'ces' or sw_ == 'czech'):
                    name = 'Stopwords-Czech.txt'
                elif (sw_ == 'en' or sw_ == 'eng' or sw_ == 'english'):
                    name = 'Stopwords-English.txt'
                elif (sw_ == 'fi' or sw_ == 'fin' or sw_ == 'finnish'):
                    name = 'Stopwords-Finnish.txt'
                elif (sw_ == 'fr' or sw_ == 'fre' or sw_ == 'fra' or sw_ == 'french'):
                    name = 'Stopwords-French.txt'
                elif (sw_ == 'de' or sw_ == 'ger' or sw_ == 'deu' or sw_ == 'german'):
                    name = 'Stopwords-German.txt'
                elif (sw_ == 'el' or sw_ == 'gre' or sw_ == 'greek'):
                    name = 'Stopwords-Greek.txt'
                elif (sw_ == 'he' or sw_ == 'heb' or sw_ == 'hebrew'):
                    name = 'Stopwords-Hebrew.txt'
                elif (sw_ == 'hi' or sw_ == 'hin' or sw_ == 'hind'):
                    name = 'Stopwords-Hind.txt'
                elif (sw_ == 'hu' or sw_ == 'hun' or sw_ == 'hungarian'):
                    name = 'Stopwords-Hungarian.txt'
                elif (sw_ == 'it' or sw_ == 'ita' or sw_ == 'italian'):
                    name = 'Stopwords-Italian.txt'
                elif (sw_ == 'ja' or sw_ == 'jpn' or sw_ == 'japanese'):
                    name = 'Stopwords-Japanese.txt'
                elif (sw_ == 'ko' or sw_ == 'kor' or sw_ == 'korean'):
                    name = 'Stopwords-Korean.txt'
                elif (sw_ == 'mr' or sw_ == 'mar' or sw_ == 'marathi'):
                    name = 'Stopwords-Marathi.txt'
                elif (sw_ == 'fa' or sw_ == 'per' or sw_ == 'fas' or sw_ == 'persian'):
                    name = 'Stopwords-Persian.txt'
                elif (sw_ == 'pl' or sw_ == 'pol' or sw_ == 'polish'):
                    name = 'Stopwords-Polish.txt'
                elif (sw_ == 'pt-br' or sw_ == 'por-br' or sw_ == 'portuguese-br'):
                    name = 'Stopwords-Portuguese-br.txt'
                elif (sw_ == 'ro' or sw_ == 'rum' or sw_ == 'ron' or sw_ == 'romanian'):
                    name = 'Stopwords-Romanian.txt'
                elif (sw_ == 'ru' or sw_ == 'rus' or sw_ == 'russian'):
                    name = 'Stopwords-Russian.txt'
                elif (sw_ == 'sk' or sw_ == 'slo' or sw_ == 'slovak'):
                    name = 'Stopwords-Slovak.txt'
                elif (sw_ == 'es' or sw_ == 'spa' or sw_ == 'spanish'):
                    name = 'Stopwords-Spanish.txt'
                elif (sw_ == 'sv' or sw_ == 'swe' or sw_ == 'swedish'):
                    name = 'Stopwords-Swedish.txt'
                elif (sw_ == 'th' or sw_ == 'tha' or sw_ == 'thai'):
                    name = 'Stopwords-Thai.txt'
                elif (sw_ == 'uk' or sw_ == 'ukr' or sw_ == 'ukrainian'):
                    name = 'Stopwords-Ukrainian.txt'
                with pkg_resources.open_binary(stws, name) as file:
                    raw_data = file.read()
                result   = chardet.detect(raw_data)
                encoding = result['encoding']
                with pkg_resources.open_text(stws, name, encoding = encoding) as file:
                    content = file.read().split('\n')
                content = [line.rstrip('\r').rstrip('\n') for line in content]
                sw      = list(filter(None, content))
                sw_full.extend(sw)
        if (len(rmv_custom_words) > 0):
            sw_full.extend(rmv_custom_words)
        try:
            vec = CountVectorizer(stop_words = frozenset(sw_full), ngram_range = (ngrams, ngrams)).fit(corpora)
        except: 
            vec = CountVectorizer(stop_words = sw_full, ngram_range = (ngrams, ngrams)).fit(corpora)
        bag_of_words = vec.transform(corpora)
        sum_words    = bag_of_words.sum(axis = 0)
        words_freq   = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq   = sorted(words_freq, key = lambda x: x[1], reverse = True)
        common_words = words_freq[:wordsn]
        words        = []
        freqs        = []
        for word, freq in common_words:
            words.append(word)
            freqs.append(freq) 
        df              = pd.DataFrame({'Word': words, 'Freq': freqs})
        self.ask_gpt_ng = pd.DataFrame({'Word': words, 'Freq': freqs})
        fig             = go.Figure(go.Bar(
                                            x           = df['Freq'],
                                            y           = df['Word'],
                                            orientation = 'h',
                                            marker      = dict(color = 'rgba(246, 78, 139, 0.6)', line = dict(color = 'black', width = 1))
                                           ),
                                    )
        fig.update_yaxes(autorange = 'reversed')
        fig.update_layout(paper_bgcolor = 'rgb(248, 248, 255)', plot_bgcolor = 'rgb(248, 248, 255)')
        fig.show()
        return 
    
    # Function: Tree Map
    def tree_map(self, view = 'browser', entry = 'kwp', topn = 20):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if   (entry == 'kwp'):
            labels = self.u_kid
            sizes  = self.kid_count 
            title  = 'Keywords Plus'
        elif (entry == 'kwa'):
            labels = self.u_auk
            sizes  = self.auk_count 
            title  = "Authors' Keywords"
        elif (entry == 'aut'):
            labels = self.u_aut
            sizes  = self.doc_aut
            title  = 'Authors'
        elif (entry == 'jou'):
            labels = self.u_jou
            sizes  = self.jou_count
            title  = 'Sources'
        elif (entry == 'ctr'):
            labels = self.u_ctr
            sizes  = self.ctr_count 
            title  = 'Countries'
        elif (entry == 'inst'):
            labels = self.u_uni
            sizes  = self.uni_count 
            title  = 'Institutions'
        idx            = sorted(range(len(sizes)), key = sizes.__getitem__)
        idx.reverse()
        labels         = [labels[i] for i in idx]
        sizes          = [sizes[i]  for i in idx]
        labels         = labels[:topn]
        sizes          = sizes[:topn]
        display_labels = []
        for lbl in labels:
            if (len(lbl) > 20):
                midpoint  = len(lbl)//2
                break_pos = lbl.find(' ', midpoint)
                if (break_pos == -1):
                    break_pos = midpoint
                lbl_display = lbl[:break_pos] + '<br>' + lbl[break_pos+1:]
            else:
                lbl_display = lbl
            display_labels.append(lbl_display)
        fig = go.Figure(
                        go.Treemap(
                                    labels        = display_labels,
                                    parents       = [''] * len(labels),  
                                    values        = sizes,
                                    textinfo      = 'label+value',  
                                    texttemplate  = '%{label}<br>(%{value})',
                                    textfont      = dict(size = 12),
                                    marker        = dict(colors = self.color_names)
                                 )
                       )
        fig.update_layout(title = title, margin = dict(l = 10, r = 10, t = 40, b = 10), paper_bgcolor = 'white')
        fig.show()
        return
    
    # Function: Authors' Productivity Plot
    def authors_productivity(self, view = 'browser', topn = 20): 
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if (topn > len(self.u_aut)):
            topn = len(self.u_aut)
        years        = list(range(self.date_str, self.date_end+1))    
        dicty        = dict(zip(years, list(range(0, len(years)))))
        idx          = sorted(range(0, len(self.doc_aut)), key = self.doc_aut.__getitem__)
        idx.reverse()
        key          = [self.u_aut[i] for i in idx if self.u_aut[i].strip().lower() != 'unknown']
        key          = key[:topn]
        n_id         = [[ [] for item in years] for item in key]
        productivity = pd.DataFrame(np.zeros((topn, len(years))), index = key, columns = years)
        Xv           = []
        Yv           = []
        Xe           = []
        Ye           = []
        for n in range(0, len(key)):
            name = key[n]
            docs = [i for i in range(0, len(self.aut)) if name in self.aut[i]]
            for i in docs:
                j                       = dicty[int(self.data.loc[i, 'year'])]
                productivity.iloc[n, j] = productivity.iloc[n, j] + 1
                n_id[n][j].append( 'id: '+str(i)+' ('+name+', '+self.data.loc[i, 'year']+')')
                Xv.append(n)
                Yv.append(j)
        self.ask_gpt_ap = productivity.copy(deep = True)
        self.ask_gpt_ap = self.ask_gpt_ap.loc[(self.ask_gpt_ap.sum(axis = 1) != 0), (self.ask_gpt_ap.sum(axis = 0) != 0)]
        node_list_a     = [ str(int(productivity.iloc[Xv[i], Yv[i]])) for i in range(0, len(Xv)) ]
        nid_list        = [ n_id[Xv[i]][Yv[i]] for i in range(0, len(Xv)) ]
        nid_list_a      = []
        for item in nid_list:
            if (len(item) == 1):
                nid_list_a.append(item)
            else:
                itens = []
                itens.append(item[0])
                for i in range(1, len(item)):
                    itens[0] = itens[0]+'<br>'+item[i]
                nid_list_a.append(itens)
        nid_list_a = [txt[0] for txt in nid_list_a]
        for i in range(0, len(Xv)-1):
            if (Xv[i] == Xv[i+1]):
                Xe.append(Xv[i]*1.00)
                Xe.append(Xv[i+1]*1.00)
                Xe.append(None)
                Ye.append(Yv[i]*1.00)
                Ye.append(Yv[i+1]*1.00)
                Ye.append(None)
        a_trace = go.Scatter(x         = Ye,
                             y         = Xe,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(255, 0, 0, 1)', width = 1.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = Yv,
                             y         = Xv,
                             opacity   = 1,
                             mode      = 'markers+text',
                             marker    = dict(symbol = 'circle-dot', size = 25, color = 'purple'),
                             text      = node_list_a,
                             hoverinfo = 'text',
                             hovertext = nid_list_a,
                             name      = ''
                             )
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = '#e0e0e0',
                            xaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True, 
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, len(years))),
                                                  ticktext       = years,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                               ),
                            yaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True,
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, topn)),
                                                  ticktext       = key,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                                )
                            )
        fig_aut = go.Figure(data = [a_trace, n_trace], layout = layout)
        fig_aut.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_aut.update_yaxes(autorange = 'reversed')
        fig_aut.show() 
        return 

    # Function: Countries' Productivity Plot
    def countries_productivity(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        years       = list(range(self.date_str, self.date_end + 1))
        yearly_data = {year: {} for year in years}
        for i, year in enumerate(self.data['year']):
            if (int(year) in years):
                countries = [item.lower() for item in self.ctr[i]]
                for country in countries:
                    if (country != 'unknown'):
                        if (country not in yearly_data[int(year)]):
                            yearly_data[int(year)][country] = 0
                        yearly_data[int(year)][country] = yearly_data[int(year)][country] + 1
        all_years_data = {}
        for year in yearly_data:
            for country, count in yearly_data[year].items():
                if (country not in all_years_data):
                    all_years_data[country] = 0
                all_years_data[country] = all_years_data[country] + count
        yearly_data['All Years'] = all_years_data
        self.ask_gpt_cp          = pd.DataFrame(yearly_data)
        self.ask_gpt_cp          = self.ask_gpt_cp.fillna(0).astype(int)
        self.ask_gpt_cp['total'] = self.ask_gpt_cp.sum(axis = 1)
        self.ask_gpt_cp          = self.ask_gpt_cp.sort_values(by = 'total', ascending = False)
        self.ask_gpt_cp          = self.ask_gpt_cp.drop(columns = 'total')
        frames                   = []
        for year in yearly_data:
            frame_data = yearly_data[year]
            countries  = list(frame_data.keys())
            counts     = list(frame_data.values())
            frames.append(go.Frame(
                                    data=[go.Choropleth(
                                                        locations        = countries,
                                                        locationmode     = 'country names',
                                                        z                = counts,
                                                        colorscale        = 'sunsetdark',
                                                        marker_line_color = 'black',
                                                        marker_line_width = 0.5,
                                                        colorbar_title    = 'Frequency',
                                                        hoverinfo         = 'location+z'
                                    )   ],
                name    = str(year)
            ))
            
        initial_data = go.Choropleth(
                                        locations         = list(all_years_data.keys()),
                                        locationmode      = 'country names',
                                        z                 = list(all_years_data.values()),
                                        colorscale        = 'sunsetdark',
                                        marker_line_color = 'black',
                                        marker_line_width = 0.5,
                                        colorbar_title    = 'Frequency',
                                        hoverinfo         = 'location+z'
        )
    
        layout = go.Layout(
                            geo = dict(
                                        scope          = 'world',
                                        showcoastlines = True,
                                        coastlinecolor = 'black',
                                        showland       = True,
                                        landcolor      = '#f0f0f0',
                                        showocean      = False,
                                        oceancolor     = '#7fcdff',
                                        showlakes      = False,
                                        lakecolor      = 'blue',
                                        showrivers     = False,
                                        resolution     = 50,
                                        lataxis        = dict(range = [-60, 90]),
                    ),
            showlegend  = False,
            hovermode   = 'closest',
            margin      = dict(b = 10, l = 5, r = 5, t = 10),
            updatemenus = [
                dict(
                    type       = 'buttons',
                    showactive = False,
                    buttons    = [
                                    dict(label = 'Play', method = 'animate', args = [None, dict(frame = dict(duration = 500, redraw = True), fromcurrent = True)]),
                                    dict(label = 'Pause', method = 'animate', args = [[None], dict(frame = dict(duration = 0, redraw = False), mode = 'immediate', transition = dict(duration = 0))])
                                ]
                            )
                        ],
            sliders = [
                dict(
                    active = len(years),
                    pad    = {'t': 50},
                    steps  = [
                                dict(
                                        label  = str(year),
                                        method = 'animate',
                                        args   = [[str(year)], dict(mode = 'immediate', frame = dict(duration = 500, redraw = True), transition = dict(duration = 300))]
                                    ) for year in yearly_data
                    ]
                )
            ]
        )
        fig = go.Figure(data = [initial_data], layout = layout, frames = frames)
        fig.show()
        return

    # Function: Institutions' Productivity Plot
    def institution_productivity(self, view = 'browser', topn = 20): 
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if (topn > len(self.u_uni)):
            topn = len(self.u_uni)
        years        = list(range(self.date_str, self.date_end+1))    
        dicty        = dict(zip(years, list(range(0, len(years)))))
        idx          = sorted(range(0, len(self.uni_count)), key = self.uni_count.__getitem__)
        idx.reverse()
        key          = [self.u_uni[i] for i in idx if self.u_uni[i].strip().lower() != 'unknown']
        key          = key[:topn]
        n_id         = [[ [] for item in years] for item in key]
        productivity = pd.DataFrame(np.zeros((topn, len(years))), index = key, columns = years)
        Xv           = []
        Yv           = []
        Xe           = []
        Ye           = []
        for n in range(0, len(key)):
            name = key[n]
            docs = [i for i in range(0, len(self.uni)) if name in self.uni[i]]
            for i in docs:
                j                       = dicty[int(self.data.loc[i, 'year'])]
                productivity.iloc[n, j] = productivity.iloc[n, j] + 1
                n_id[n][j].append( 'id: '+str(i)+' ('+name+', '+self.data.loc[i, 'year']+')')
                Xv.append(n)
                Yv.append(j)
        self.ask_gpt_ip = productivity.copy(deep = True)
        self.ask_gpt_ip = self.ask_gpt_ip.loc[(self.ask_gpt_ip.sum(axis = 1) != 0), (self.ask_gpt_ip.sum(axis = 0) != 0)]
        node_list_a     = [ str(int(productivity.iloc[Xv[i], Yv[i]])) for i in range(0, len(Xv)) ]
        nid_list        = [ n_id[Xv[i]][Yv[i]] for i in range(0, len(Xv)) ]
        nid_list_a      = []
        for item in nid_list:
            if (len(item) == 1):
                nid_list_a.append(item)
            else:
                itens = []
                itens.append(item[0])
                for i in range(1, len(item)):
                    itens[0] = itens[0]+'<br>'+item[i]
                nid_list_a.append(itens)
        nid_list_a = [txt[0] for txt in nid_list_a]
        for i in range(0, len(Xv)-1):
            if (Xv[i] == Xv[i+1]):
                Xe.append(Xv[i]*1.00)
                Xe.append(Xv[i+1]*1.00)
                Xe.append(None)
                Ye.append(Yv[i]*1.00)
                Ye.append(Yv[i+1]*1.00)
                Ye.append(None)
        a_trace = go.Scatter(x         = Ye,
                             y         = Xe,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(255, 0, 0, 1)', width = 1.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = Yv,
                             y         = Xv,
                             opacity   = 1,
                             mode      = 'markers+text',
                             marker    = dict(symbol = 'circle-dot', size = 25, color = '#ff7f0e'),
                             text      = node_list_a,
                             hoverinfo = 'text',
                             hovertext = nid_list_a,
                             name      = ''
                             )
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = '#e0e0e0',
                            xaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True, 
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, len(years))),
                                                  ticktext       = years,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                               ),
                            yaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True,
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, topn)),
                                                  ticktext       = key,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                                )
                            )
        fig_inst = go.Figure(data = [a_trace, n_trace], layout = layout)
        fig_inst.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_inst.update_yaxes(autorange = 'reversed')
        fig_inst.show() 
        return 

    # Function: Sources' Productivity Plot
    def source_productivity(self, view = 'browser', topn = 20): 
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if (topn > len(self.u_jou)):
            topn = len(self.u_jou)
        years        = list(range(self.date_str, self.date_end+1))    
        dicty        = dict(zip(years, list(range(0, len(years)))))
        idx          = sorted(range(0, len(self.jou_count)), key = self.jou_count.__getitem__)
        idx.reverse()
        key          = [self.u_jou[i] for i in idx if self.u_jou[i].strip().lower() != 'unknown']
        key          = key[:topn]
        n_id         = [[ [] for item in years] for item in key]
        productivity = pd.DataFrame(np.zeros((topn, len(years))), index = key, columns = years)
        Xv           = []
        Yv           = []
        Xe           = []
        Ye           = []
        for n in range(0, len(key)):
            name = key[n]
            docs = [i for i in range(0, len(self.jou)) if name in self.jou[i]]
            for i in docs:
                j                       = dicty[int(self.data.loc[i, 'year'])]
                productivity.iloc[n, j] = productivity.iloc[n, j] + 1
                n_id[n][j].append( 'id: '+str(i)+' ('+name+', '+self.data.loc[i, 'year']+')')
                Xv.append(n)
                Yv.append(j)
        self.ask_gpt_sp = productivity.copy(deep = True)
        self.ask_gpt_sp = self.ask_gpt_sp.loc[(self.ask_gpt_sp.sum(axis = 1) != 0), (self.ask_gpt_sp.sum(axis = 0) != 0)]
        node_list_a     = [ str(int(productivity.iloc[Xv[i], Yv[i]])) for i in range(0, len(Xv)) ]
        nid_list        = [ n_id[Xv[i]][Yv[i]] for i in range(0, len(Xv)) ]
        nid_list_a      = []
        for item in nid_list:
            if (len(item) == 1):
                nid_list_a.append(item)
            else:
                itens = []
                itens.append(item[0])
                for i in range(1, len(item)):
                    itens[0] = itens[0]+'<br>'+item[i]
                nid_list_a.append(itens)
        nid_list_a = [txt[0] for txt in nid_list_a]
        for i in range(0, len(Xv)-1):
            if (Xv[i] == Xv[i+1]):
                Xe.append(Xv[i]*1.00)
                Xe.append(Xv[i+1]*1.00)
                Xe.append(None)
                Ye.append(Yv[i]*1.00)
                Ye.append(Yv[i+1]*1.00)
                Ye.append(None)
        a_trace = go.Scatter(x         = Ye,
                             y         = Xe,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(255, 0, 0, 1)', width = 1.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = Yv,
                             y         = Xv,
                             opacity   = 1,
                             mode      = 'markers+text',
                             marker    = dict(symbol = 'circle-dot', size = 25, color = 'green'),
                             text      = node_list_a,
                             hoverinfo = 'text',
                             hovertext = nid_list_a,
                             name      = ''
                             )
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = '#e0e0e0',
                            xaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True, 
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, len(years))),
                                                  ticktext       = years,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                               ),
                            yaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True,
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, topn)),
                                                  ticktext       = key,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                                )
                            )
        fig_inst = go.Figure(data = [a_trace, n_trace], layout = layout)
        fig_inst.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_inst.update_yaxes(autorange = 'reversed')
        fig_inst.show() 
        return

    # Function: Evolution per Year
    def plot_evolution_year(self, view = 'browser', stop_words = ['en'], key = 'kwp', rmv_custom_words = [], topn = 10, txt_font_size = 10, start = 2010, end = 2022):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if (start < self.date_str or start == -1):
            start = self.date_str
        if (end > self.date_end or end == -1):
            end = self.date_end
        y_idx = [i for i in range(0, self.data.shape[0]) if int(self.data.loc[i, 'year']) >= start and int(self.data.loc[i, 'year']) <= end]
        if (len(rmv_custom_words) == 0):
            rmv_custom_words = ['unknown']
        else:
            rmv_custom_words.append('unknown') 
        if   (key == 'kwp'):
            u_ent = [item for item in self.u_kid if item not in rmv_custom_words]
            ent   = [item for item in self.kid   if item not in rmv_custom_words]
        elif (key == 'kwa'):
            u_ent = [item for item in self.u_auk if item not in rmv_custom_words]
            ent   = [item for item in self.auk   if item not in rmv_custom_words]
        elif (key == 'jou'):
            u_ent = [item for item in self.u_jou if item not in rmv_custom_words]
            ent   = [item for item in self.jou   if item not in rmv_custom_words]
        elif (key == 'abs'):
            abs_  = self.data['abstract'].tolist()
            abs_  = ['the' if i not in y_idx else  abs_[i] for i in range(0, len(abs_))]
            abs_  = self.clear_text(abs_, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
            u_abs = [item.split() for item in abs_]
            u_abs = [item for sublist in u_abs for item in sublist]
            u_abs = list(set(u_abs))
            if (u_abs[0] == ''):
                u_abs = u_abs[1:]
            s_abs       = [item.split() for item in abs_]
            s_abs       = [item for sublist in s_abs for item in sublist]
            abs_count   = [s_abs.count(item) for item in u_abs]
            idx         = sorted(range(len(abs_count)), key = abs_count.__getitem__)
            idx.reverse()
            abs_       = [item.split() for item in abs_]
            u_abs      = [u_abs[i] for i in idx]
            u_ent, ent = u_abs, abs_
        elif (key == 'title'):
            tit_  = self.data['title'].tolist()
            tit_  = ['the' if i not in y_idx else  tit_[i] for i in range(0, len(tit_))]
            tit_  = self.clear_text(tit_, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
            u_tit = [item.split() for item in tit_]
            u_tit = [item for sublist in u_tit for item in sublist]
            u_tit = list(set(u_tit))
            if (u_tit[0] == ''):
                u_tit = u_tit[1:]
            s_tit       = [item.split() for item in tit_]
            s_tit       = [item for sublist in s_tit for item in sublist]
            tit_count   = [s_tit.count(item) for item in u_tit]
            idx         = sorted(range(len(tit_count)), key = tit_count.__getitem__)
            idx.reverse()
            tit_       = [item.split() for item in tit_]
            u_tit      = [u_tit[i] for i in idx]
            u_ent, ent = u_tit, tit_
        traces          = []
        years           = list(range(self.date_str, self.date_end+1))
        dict_y          = dict(zip(years, list(range(0, len(years)))))
        themes          = self.__get_counts_year(u_ent, ent)
        self.ask_gpt_ep = ''
        for j in range(dict_y[start], dict_y[end]+1):
            theme_vec = themes.iloc[:, j]
            theme_vec = theme_vec[theme_vec > 0]
            if (len(theme_vec) > 0):
                theme_vec       = theme_vec.sort_values(ascending = False) 
                theme_vec       = theme_vec.iloc[:topn] 
                idx             = theme_vec.index.tolist()
                names           = [u_ent[item] for item in idx]
                values          = [themes.loc[item, j] for item in idx]
                n_val           = [names[i]+' ('+str(int(values[i]))+')' for i in range(0, len(names))]
                self.ask_gpt_ep = self.ask_gpt_ep + ' ' + str(years[j]) + ': ' + ', '.join(n_val)
                data            = go.Bar(x                = [years[j]]*len(values), 
                                         y                = values, 
                                         text             = names, 
                                         hoverinfo        = 'text',
                                         textangle        = 0,
                                         textfont_size    = txt_font_size,
                                         hovertext        = n_val,
                                         insidetextanchor = 'middle',
                                         marker_color     = self.__hex_rgba(hxc = self.color_names[j], alpha = 0.70)
                                         )
                traces.append(data)
        layout = go.Layout(barmode      = 'stack', 
                           showlegend   = False,
                           hovermode    = 'closest',
                           margin       = dict(b = 10, l = 5, r = 5, t = 10),
                           plot_bgcolor = '#f5f5f5',
                           xaxis        = dict(tickangle      =  35,
                                               showticklabels = True, 
                                               type           = 'category'
                                              )
                           )
        fig = go.Figure(data = traces, layout = layout)
        fig.show()
        return     
    
    # Function: Parse Evolution Plot Data
    def parse_ep_data(self, input_text):
        data         = defaultdict(dict)
        year_entries = re.split(r'(\d{4}):', input_text)[1:]  
        years        = year_entries[::2] 
        entries      = year_entries[1::2] 
        for year, entry in zip(years, entries):
            items = re.findall(r'([^,]+?)\s\((\d+)\)', entry)
            for keyword, count in items:
                data[year.strip()][keyword.strip()] = int(count)
        return dict(data)
    
    # Function: Evolution per Year Complement
    def plot_evolution_year_complement(self, ep_text, view = 'browser', topn = 10, custom = []):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        data = self.parse_ep_data(ep_text)
        df   = pd.DataFrame(data).fillna(0)
        df   = df.T
        if (len(custom) > 0):
            df = df[custom]
        elif topn is not None:
            total_frequencies = df.sum(axis = 0).sort_values(ascending = False)
            top_keywords      = total_frequencies.index[:topn]
            df                = df[top_keywords]
        fig    = go.Figure()
        for idx, keyword in enumerate(df.columns):
            fig.add_trace(go.Scatter(
                                        x          = df.index,
                                        y          = df[keyword],
                                        mode       = 'lines',
                                        stackgroup = 'one', 
                                        name       = keyword,
                                        line       = dict(width = 0.5, color = self.color_names[idx]), 
                                        hoverinfo  = 'x+y+name'  
                                    ))
    
        fig.update_layout(
                            hovermode   = 'x unified', 
                            showlegend  = True,
                            legend      = dict(title = 'Keywords'),
                            xaxis       = dict(
                                                title             = 'Year',
                                                tickmode          = 'array',
                                                tickvals          = list(df.index),
                                                ticktext          = list(df.index),
                                                ticklabelposition = 'outside',  
                                                tickangle         = -45,  
                                                automargin        = True  
                                            ),
                            yaxis       = dict(
                                                title             = 'Frequency'
                                            )
                        )
        fig.show()
        return

    # Function: Plot Bar 
    def plot_bars(self, view = 'browser',  statistic = 'dpy', topn = 20):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if  (statistic.lower() == 'dpy'):
            key   = list(range(self.date_str, self.date_end+1))
            value = [self.data[self.data.year == str(item)].shape[0] for item in key]
            title = 'Documents per Year'
            x_lbl = 'Year'
            y_lbl = 'Documents'
        elif(statistic.lower() == 'cpy'):
            key   = list(range(self.date_str, self.date_end+1))
            value = []
            title = 'Citations per Year'
            x_lbl = 'Year'
            y_lbl = 'Citations'
            for i in range(0, len(key)):
                year = key[i]
                idx  = [i for i, x in enumerate(list(self.dy)) if x == year]
                docs = 0
                for j in idx:
                    docs = docs + self.citation[j]
                value.append(docs)
        elif(statistic.lower() == 'ppy'):
            key, value = self.__get_past_citations_year()
            title = 'Past Citations per Year'
            x_lbl = 'Year'
            y_lbl = 'Past Citations'
        elif (statistic.lower() == 'ltk'):
            value = list(range(1, max(self.doc_aut)+1))
            key   = [self.doc_aut.count(item) for item in value]
            idx   = [i for i in range(0, len(key)) if key[i] > 0]
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            accumulator = defaultdict(int)
            for k, v in zip(key, value):
                accumulator[k] = accumulator[k] + v
            combined   = sorted(accumulator.items(), key = lambda x: x[0])
            key, value = zip(*combined)
            key        = [item for item in key]
            value      = [item for item in value]
            title      = "Lotka's Law"
            x_lbl      = 'Documents'
            y_lbl      = 'Authors'
        elif (statistic.lower() == 'spd'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Sources per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'spc'):
            key   = self.u_jou
            value = self.jou_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Sources per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'apd'):
            key   = self.u_aut
            value = self.doc_aut
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Authors per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'apc'):
            key   = self.u_aut
            value = self.aut_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Authors per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'aph'):
            key   = self.u_aut
            value = self.aut_h
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Authors per H-Index'
            x_lbl = 'H-Index'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'bdf_1'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            key   = [key[i]   for i in range(0, len(key))   if value[i] <= c1]
            value = [value[i] for i in range(0, len(value)) if value[i] <= c1]
            title = "Bradford's Law - Core Sources 1"
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'bdf_2'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            c2    = int(value[-1]*(2/3))
            key   = [key[i]   for i in range(0, len(key))   if value[i] > c1 and value[i] <= c2]
            value = [value[i] for i in range(0, len(value)) if value[i] > c1 and value[i] <= c2]
            title = "Bradford's Law - Core Sources 2"
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'bdf_3'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c2    = int(value[-1]*(2/3))
            key   = [key[i]   for i in range(0, len(key))   if value[i] > c2]
            value = [value[i] for i in range(0, len(value)) if value[i] > c2]
            title = "Bradford's Law - Core Sources 3"
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'ipd'):
            key   = self.u_uni
            value = self.uni_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Institutions per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Institutions'
        elif (statistic.lower() == 'ipc'):
            key   = self.u_uni
            value = self.uni_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Institutions per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Institutions'
        elif (statistic.lower() == 'cpd'):
            key   = self.u_ctr
            value = self.ctr_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Countries per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Countries'
        elif (statistic.lower() == 'cpc'):
            key   = self.u_ctr
            value = self.ctr_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Countries per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Countries'
        elif (statistic.lower() == 'lpd'):
            key   = self.u_lan
            value = self.lan_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Language per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Languages'
        elif (statistic.lower() == 'kpd'):
            key   = self.u_kid
            value = self.kid_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Keywords Plus per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Keywords Plus'
        elif (statistic.lower() == 'kad'):
            key   = self.u_kid
            value = self.kid_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+" - Authors' Keywords per Documents"
            x_lbl = 'Documents'
            y_lbl = "Authors' Keywords"
        data_tuples       = list(zip(key, value))
        self.ask_gpt_bp   = pd.DataFrame(data_tuples, columns = [x_lbl, y_lbl])
        self.ask_gpt_bp_t = title
        fig = go.Figure()
        if (statistic.lower() in ['dpy', 'cpy', 'ppy', 'ltk']):
            x = key
            y = value
            fig.add_trace(go.Bar(
                            x            = x, 
                            y            = y, 
                            text         = y,
                            textposition = 'auto',
                            marker       = dict(color = '#ffc001', line = dict(color = '#c3d5e8', width = 1)),
                          ))
            
            if (statistic.lower() != 'ltk'):
                avg_line = np.mean(value)
                fig.add_hline(y = avg_line, line_color = 'red', line_width = 1, line_dash = 'solid')  
            fig.update_layout(title = title, xaxis_title = x_lbl, yaxis_title = y_lbl, template = 'plotly_white')
        else:
            x = value
            y = key
            fig.add_trace(go.Bar(
                            x            = x,
                            y            = y,
                            orientation  = 'h',
                            text         = x,
                            textposition = 'auto',
                            marker       = dict(color = '#ffc001', line = dict(color = '#c3d5e8', width = 1))
                        ))
            fig.update_yaxes(autorange = 'reversed')
            fig.update_layout(title = title, xaxis_title = x_lbl, yaxis_title = y_lbl, template = 'plotly_white')
        if (statistic.lower() == 'ltk'):
            fig.update_xaxes(type = 'category', categoryorder = 'array', categoryarray = x)
        fig.show()
        return
    
    # Function: Enumerate Relationship
    def enumerate_relationships(self, sk_data, entry, rmv_unknowns):
        rel_lists = []  
        for i in range(0, len(entry) - 1):
            source_col = entry[i]
            target_col = entry[i + 1]
            rel         = []
            for _, row in sk_data.iterrows():
                sources, targets = row[source_col], row[target_col]
                if (len(sources) == len(targets)):
                    pairs = list(zip(sources, targets)) 
                else:
                    pairs = [(s, t) for s in sources for t in targets]  
                updated_pairs = []
                for a, b in pairs:
                    a_unknown = 'unknown' in str(a).lower()
                    b_unknown = 'unknown' in str(b).lower()
                    if (rmv_unknowns == True):
                        if not a_unknown and not b_unknown:
                            updated_pairs.append((a, b))
                    else:
                        if a_unknown and b_unknown:
                            updated_pairs.append((f"UNKNOWN_{source_col}", f"UNKNOWN_{target_col}"))
                        elif a_unknown:
                            updated_pairs.append((f"UNKNOWN_{source_col}", b))
                        elif b_unknown:
                            updated_pairs.append((a, f"UNKNOWN_{target_col}"))
                        else:
                            updated_pairs.append((a, b))
                rel.extend(updated_pairs)  
            rel_lists.append(rel)  
        return rel_lists
        
    # Function: Plot Y per X
    def plot_count_y_per_x(self, view = 'browser', rmv_unknowns = True, x = 'cout', y = 'aut', topn_x = 5, topn_y = 5, text_font_size = 12, x_angle = -90):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        sk_data = pd.DataFrame({ 'aut' : self.aut,
                                 'cout': self.ctr,
                                 'inst': self.uni,
                                 'jou' : self.jou,
                                 'kwa' : self.auk,
                                 'kwp' : self.kid,
                                 'lan' : self.lan})
        sk_data              = sk_data[[x, y]]        
        relationships        = self.enumerate_relationships(sk_data, [x, y], rmv_unknowns)
        x_y_pairs            = relationships[0]
        y_counts             = Counter(x_y_pairs)
        y_df                 = pd.DataFrame(y_counts.items(), columns = ['Pair', 'Count'])
        y_df[['X', 'Y']]     = pd.DataFrame(y_df['Pair'].tolist(), index = y_df.index)
        y_df.drop(columns    = ['Pair'], inplace = True)
        x_counts             = y_df.groupby('X')['Count'].sum().reset_index()
        top_x                = x_counts.nlargest(topn_x, 'Count')['X'].tolist()
        filtered_df          = y_df[y_df['X'].isin(top_x)]
        filtered_df          = filtered_df.sort_values(['X', 'Count'], ascending = [True, False])
        top_y_df             = filtered_df.groupby('X').head(topn_y)
        self.top_y_x         = top_y_df.copy()
        u_keys               = ['aut', 'cout', 'inst', 'jou', 'kwa', 'kwp', 'lan']
        u_name               = ['Authors', 'Countries', 'Institutions', 'Journals', 'Auhors_Keywords', 'Keywords_Plus', 'Languages']
        dict_n               = dict( zip( u_keys, u_name ) )
        self.top_y_x.columns = ['Count', 'X ('+str(dict_n[x])+')', 'Y ('+str(dict_n[y])+')']
        fig                  = go.Figure()
        for _, row in top_y_df.iterrows():
            y_text = f"{row['Y']} ({row['Count']})" 
            fig.add_trace(go.Bar(
                x            = [row['X']], 
                y            = [row['Count']], 
                name         = row['Y'],  
                text         = y_text, 
                textposition = 'inside',
                textfont     = dict(size = text_font_size), 
                hoverinfo    = "x+y+text", 
            ))
        fig.update_layout(
            title       = f"Distribution of {str(dict_n[y]) } per {str(dict_n[x]) }",
            xaxis_title = str(dict_n[x]) ,
            yaxis_title = str(dict_n[y]) ,
            barmode     = 'stack',  
            template    = 'plotly',
            showlegend  = False,
            xaxis       = dict( categoryorder = 'total descending',  tickangle = x_angle )
            )
        fig.show()
        return

    # Function: Heatmap Plot Y per X
    def plot_heatmap_y_per_x(self, x, y, topn_x = 5, topn_y = 5, element_x = [], element_y = [], rmv_unknowns = True, view = "browser"):
        if view == "browser":
            pio.renderers.default = "browser"
        sk_data = pd.DataFrame({ 'aut' : self.aut,
                                 'cout': self.ctr,
                                 'inst': self.uni,
                                 'jou' : self.jou,
                                 'kwa' : self.auk,
                                 'kwp' : self.kid,
                                 'lan' : self.lan})
        sort_x      = True
        sort_y      = True
        sk_data     = sk_data[[x, y]].copy()
        pair_to_ids = defaultdict(list)
        for doc_id, row in sk_data.iterrows():
            xs = row[x] if isinstance(row[x], (list, tuple, set)) else [row[x]]
            ys = row[y] if isinstance(row[y], (list, tuple, set)) else [row[y]]
            for xi in xs:
                for yi in ys:
                    if rmv_unknowns and (str(xi).lower() == "unknown" or str(yi).lower() == "unknown"):
                        continue
                    pair_to_ids[(xi, yi)].append(doc_id)
        #x_counter = Counter()
        #for (xi, yi), id_list in pair_to_ids.items():
            #x_counter[xi] += len(id_list)
        #top_x     = [xi for xi, _ in x_counter.most_common(topn_x)]
        #y_counter = Counter()
        #for (xi, yi), id_list in pair_to_ids.items():
            #y_counter[yi] += len(id_list)
        #top_y         = [yi for yi, _ in y_counter.most_common(topn_y)]
        if len(element_x) > 0:
            sel_x = element_x
        else:
            sel_x = None
        if len(element_y) > 0:
            sel_y = element_y
        else:
            sel_y = None
        if sel_x is None:
            if sort_x:
                x_counter = Counter()
                for (xi, yi), id_list in pair_to_ids.items():
                    x_counter[xi] = x_counter[xi] + len(id_list)
                sel_x = [xi for xi, _ in x_counter.most_common(topn_x)]
            else:
                if sel_y is not None:
                    sel_x = sorted({xi for (xi, yi) in pair_to_ids if yi in sel_y})
                else:
                    sel_x = sorted({xi for (xi, yi) in pair_to_ids})   
        if sel_y is None:
            if sort_y:
                y_counter = Counter()
                for (xi, yi), id_list in pair_to_ids.items():
                    y_counter[yi] = y_counter[yi] + len(id_list)
                sel_y = [yi for yi, _ in y_counter.most_common(topn_y)]
            else:
                if sel_x is not None:
                    sel_y = sorted({yi for (xi, yi) in pair_to_ids if xi in sel_x})
                else:
                    sel_y = sorted({yi for (xi, yi) in pair_to_ids})
        matrix_ids    = []
        matrix_counts = []
        for yi in sel_y:
            row_ids    = []
            row_counts = []
            for xi in sel_x:
                ids = pair_to_ids.get((xi, yi), [])
                row_ids.append(ids.copy())         
                row_counts.append(len(ids))        
            matrix_ids.append(row_ids)
            matrix_counts.append(row_counts)
        matrix_ids    = [[list(set(cell)) if isinstance(cell,list) else cell for cell in row] for row in matrix_ids]
        matrix_counts = [[len(cell) if isinstance(cell, list) else 0 for cell in row] for row in matrix_ids]
        u_keys        = ['aut', 'cout', 'inst', 'jou', 'kwa', 'kwp', 'lan']
        u_name        = ['Authors', 'Countries', 'Institutions', 'Journals', 'Auhors_Keywords', 'Keywords_Plus', 'Languages']
        dict_n        = dict( zip( u_keys, u_name ) )
        self.heat_y_x = pd.DataFrame(matrix_ids, index = sel_y, columns = sel_x)
        hover_text    = []
        for row in matrix_ids:
            row_text = []
            for cell in row:
                if cell:
                    text = "<br>".join([f"ID: {doc_id}" for doc_id in sorted(cell)])
                else:
                    text = ""
                row_text.append(text)
            hover_text.append(row_text)
        fig = go.Figure(go.Heatmap(
                                    z             = matrix_counts,
                                    x             = sel_x,         
                                    y             = sel_y,
                                    text          = matrix_counts, 
                                    texttemplate  = "%{text}",     
                                    textfont      = dict(color = "black", size = 12),
                                    hovertext     = hover_text,
                                    hovertemplate = "%{hovertext}<extra></extra>",
                                    colorscale    = "Pinkyl",
                                    showscale     = False,
                                    xgap          = 1,
                                    ygap          = 1
                                ))
        fig.update_layout(
            title = dict(
                          text    = f"Distribution of <b>{dict_n.get(y, y)}</b> per <b>{dict_n.get(x, x)}</b>",
                          x       = 0.5,
                          xanchor = "center",
                          font    = dict(size = 20)
                        ),
            xaxis = dict(
                          title     = str(x).upper(),
                          tickangle = -45,
                          tickfont  = dict(size=12),
                          side      = "bottom",
                          type      = "category"
                        ),
            yaxis = dict(
                          title    = str(y).upper(),
                          tickfont = dict(size=12),
                          autorange = "reversed",
                          type      = "category"
                        ),
            margin = dict(t = 80, b = 80, l = 100, r = 20)
        )
        fig.show()
        return

    # Function: Sankey Diagram
    def sankey_diagram(self, view = 'browser', entry = ['aut', 'cout', 'inst', 'jou', 'kwa', 'kwp', 'lan'], rmv_unknowns = False, topn = None): 
        
        #----------------------------------------------------------------------
            
        def count_entry(sk_data, entry):
            sorted_lists = []
            for e in entry:
                flat_list     = [item for sublist in sk_data[e] for item in sublist]
                counter       = Counter(flat_list)
                sorted_counts = sorted(counter.items(), key = lambda x: x[1], reverse = True)
                sorted_lists.append(sorted_counts)
            return sorted_lists
        
        def get_top_items(pairs, target, topn = None):
            item_counts  = Counter([b for a, b in pairs if a == target])
            sorted_items = sorted(item_counts.items(), key = lambda x: x[1], reverse = True)
            return sorted_items[:topn] if topn else sorted_items

        def hierarchical_filtering(sk_data, entry, topn, rmv_unknowns):       
            if (topn is None):
                topn = [float('inf')] * (len(entry) - 1)
            if (len(topn) != len(entry) - 1):
                raise ValueError(f"topn must have length {len(entry) - 1}, received {len(topn)}.")
            count_sorted_entries   = count_entry(sk_data, entry)
            relationships          = self.enumerate_relationships(sk_data, entry, rmv_unknowns)
            filtered_relationships = {}
            prev_top_items         = [item[0] for item in count_sorted_entries[0][:topn[0]]]
            prev_level_data        = {}
            for a in prev_top_items:
                possible_targets   = [b for (x, b) in relationships[0] if x == a]  
                target_counts      = Counter(possible_targets) 
                sorted_targets     = sorted(target_counts.items(), key = lambda x: x[1], reverse = True)  
                prev_level_data[a] = [b for b, _ in sorted_targets[:topn[1]]]  
            filtered_relationships[(entry[0], entry[1])] = Counter( [(a, b) for a in prev_top_items for b in prev_level_data[a]])
            for i in range(1, len(entry) - 1):
                source_col             = entry[i]
                target_col             = entry[i + 1]
                current_relationships  = relationships[i]
                filtered_current_level = []
                for source, targets in prev_level_data.items():
                    for target in targets:
                        possible_next_targets = [new_target for (x, new_target) in current_relationships if x == target]
                        next_target_counts    = Counter(possible_next_targets)
                        sorted_next_targets   = sorted(next_target_counts.items(), key = lambda x: x[1], reverse=True)
                        top_targets           = [b for b, _ in sorted_next_targets[:topn[min(i, len(topn) - 1)]]]  
                        filtered_current_level.extend([(target, new_target) for new_target in top_targets])
                filtered_relationships[(source_col, target_col)] = Counter(filtered_current_level)
                prev_level_data                                  = {}
                for source, target in filtered_current_level:
                    if (source not in prev_level_data):
                        prev_level_data[source] = []
                    prev_level_data[source].append(target)
            return filtered_relationships, relationships 

        #----------------------------------------------------------------------
       
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        sk_data = pd.DataFrame({ 'aut' : self.aut,
                                 'cout': self.ctr,
                                 'inst': self.uni,
                                 'jou' : self.jou,
                                 'kwa' : self.auk,
                                 'kwp' : self.kid,
                                 'lan' : self.lan})
        sk_data                           = sk_data[entry]
        all_relationships, relationships  = hierarchical_filtering(sk_data, entry, topn, rmv_unknowns)
        all_pairs                         = [pair for sublist in relationships for pair in sublist]
        for (source_col, target_col), counter in all_relationships.items():
            updated_counter = Counter()  
            for (source, target) in counter.keys():
                updated_counter[(source, target)] = all_pairs.count((source, target))  
            all_relationships[(source_col, target_col)] = updated_counter
        nodes             = set()
        for (source_col, target_col), counter in all_relationships.items():
            for (source, target), count in counter.items():
                nodes.add(source)
                nodes.add(target)
        node_to_index    = {node: idx for idx, node in enumerate(nodes)}
        sk_s, sk_t, sk_v = [], [], []
        s, t, v          = [], [], []
        for (source_col, target_col), counter in all_relationships.items():
            for (source, target), count in counter.items():
                sk_s.append(node_to_index[source])
                sk_t.append(node_to_index[target])
                sk_v.append(count)
                s.append(source)
                t.append(target)
                v.append(count)
        self.ask_gpt_sk = pd.DataFrame(list(zip(s, t, v)), columns = ['Node From', 'Node To', 'Connection Weigth'])
        u_keys         = ['aut', 'cout', 'inst', 'jou', 'kwa', 'kwp', 'lan']
        u_name         = ['Authors', 'Countries', 'Institutions', 'Journals', 'Auhors_Keywords', 'Keywords_Plus', 'Languages']
        dict_n         = dict( zip( u_keys, u_name ) )
        if (len(sk_s) > len(self.color_names)):
            count = 0
            while (len(self.color_names) < len(sk_s)):
                self.color_names.append(self.color_names[count])
                count = count + 1
        link = dict(source = sk_s,        target = sk_t, value     = sk_v, color = self.color_names)
        node = dict(label  = list(nodes), pad    = 10,   thickness = 15,   color = 'white')
        data = go.Sankey(
                          link        = link, 
                          node        = node, 
                          arrangement = 'freeform'
                         )
        fig  = go.Figure(data)
        nt   = 'Sankey Diagram ( '
        for e in range(0, len(entry)):
            nt = nt + str(dict_n[entry[e]]) + ' / '
        nt = nt[:-2] + ')'
        fig.update_layout(hovermode = 'closest', title = nt, font = dict(size = 12, color = 'white'), paper_bgcolor = '#474747')
        fig.show()
        return

    #############################################################################
    
    # Function: Hirsch Index
    def h_index(self):
        h_i                     = []
        researcher_to_citations = {researcher: [] for researcher in self.u_aut}
        for i, researchers in enumerate(self.aut):
            for researcher in researchers:
                if (researcher in researcher_to_citations):
                    researcher_to_citations[researcher].append(self.citation[i])
        for researcher in self.u_aut:
            citations = researcher_to_citations[researcher]
            citations.sort(reverse = True)
            h         = 0
            for idx, citation in enumerate(citations):
                if (citation >= idx + 1):
                    h = idx + 1
                else:
                    break
            h_i.append(h)
        return h_i
    
    # Function: G-Index
    def g_index(self):
        g_i                     = []
        researcher_to_citations = {researcher: [] for researcher in self.u_aut}
        for i, authors in enumerate(self.aut):
            for researcher in authors:
                if (researcher in researcher_to_citations):
                    citation_val = self.citation[i]
                    if isinstance(citation_val, int):
                        researcher_to_citations[researcher].append(citation_val)
                    else:
                        try:
                            citation_int = int(citation_val)
                            researcher_to_citations[researcher].append(citation_int)
                        except:
                            pass
        for researcher in self.u_aut:
            citations      = researcher_to_citations[researcher]
            citations.sort(reverse = True)
            cumulative_sum = 0
            g              = 0
            for idx, citation in enumerate(citations):
                cumulative_sum = cumulative_sum + citation
                if (cumulative_sum >= (idx + 1) ** 2):
                    g = idx + 1
                else:
                    break
            g_i.append(g)
        return g_i

    # Function: M-Index
    def m_index(self, current_year):
        m_i                 = [] 
        researcher_to_years = {researcher: [] for researcher in self.u_aut}
        for i, authors in enumerate(self.aut):
            for researcher in authors:
                if (researcher in researcher_to_years):
                    year_val = self.dy[i]
                    if (year_val != -1):
                        try:
                            researcher_to_years[researcher].append(float(year_val))
                        except:
                            pass
        for idx, researcher in enumerate(self.u_aut):
            if (researcher_to_years[researcher]):
                first_year    = min(researcher_to_years[researcher])
                career_length = current_year - first_year + 1
                if (career_length < 1):
                    career_length = 1
                m_value = self.aut_h[idx] / career_length
            else:
                m_value = None
            m_i.append(m_value)
        return m_i
    
    # Function: E-Index    
    def e_index(self):
        researcher_to_citations = {researcher: [] for researcher in self.u_aut}
        for i, authors in enumerate(self.aut):
            for researcher in authors:
                if (researcher in researcher_to_citations):
                    citation_val = self.citation[i]
                    if isinstance(citation_val, int):
                        researcher_to_citations[researcher].append(citation_val)
                    else:
                        try:
                            citation_int = int(citation_val)
                            researcher_to_citations[researcher].append(citation_int)
                        except:
                            pass
        e_indices = []
        for researcher, h in zip(self.u_aut, self.aut_h):
            citations  = researcher_to_citations[researcher]
            citations.sort(reverse = True)
            excess_sum = 0
            for i in range(h):
                if (i < len(citations)):
                    excess = citations[i] - h
                    if (excess > 0):
                        excess_sum = excess_sum + excess
            e_value = np.sqrt(excess_sum)
            e_indices.append(e_value)
        return e_indices

    # Function: Total and Self Citations
    def __total_and_self_citations(self):
        preprocessed_refs = [ [ref.lower() for ref in refs] for refs in self.ref ]
        t_c               = []
        s_c               = []
        for researcher in self.u_aut:
            authored_papers = self.author_to_papers.get(researcher, [])
            if not authored_papers:
                t_c.append(0)
                s_c.append(0)
                continue
            total_citations  = sum(self.citation[paper_idx] for paper_idx in authored_papers)
            researcher_lower = researcher.lower()
            self_citations   = sum(1 for paper_idx in authored_papers for ref in preprocessed_refs[paper_idx] if researcher_lower in ref)
            t_c.append(total_citations)
            s_c.append(self_citations)
        return t_c, s_c

    #############################################################################

    # Function: Text Pre-Processing
    def clear_text(self, corpus, stop_words = ['en'], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [], verbose = False):
        sw_full = []
        if (lowercase == True):
            if (verbose == True):
                print('Lower Case: Working...')
            corpus = [str(x).lower().replace("’","'") for x in  corpus]
            if (verbose == True):
                print('Lower Case: Done!')
        if (rmv_special_chars == True):
            if (verbose == True):
                print('Removing Special Characters: Working...')
            corpus = [re.sub(r"[^a-zA-Z0-9']+", ' ', i) for i in corpus]
            if (verbose == True):
                print('Removing Special Characters: Done!')
        if (len(stop_words) > 0):
            for sw_ in stop_words: 
                if   (sw_ == 'ar' or sw_ == 'ara' or sw_ == 'arabic'):
                    name = 'Stopwords-Arabic.txt'
                elif (sw_ == 'bn' or sw_ == 'ben' or sw_ == 'bengali'):
                    name = 'Stopwords-Bengali.txt'
                elif (sw_ == 'bg' or sw_ == 'bul' or sw_ == 'bulgarian'):
                    name = 'Stopwords-Bulgarian.txt'
                elif (sw_ == 'zh' or sw_ == 'chi' or sw_ == 'chinese'):
                    name = 'Stopwords-Chinese.txt'
                elif (sw_ == 'cs' or sw_ == 'cze' or sw_ == 'ces' or sw_ == 'czech'):
                    name = 'Stopwords-Czech.txt'
                elif (sw_ == 'en' or sw_ == 'eng' or sw_ == 'english'):
                    name = 'Stopwords-English.txt'
                elif (sw_ == 'fi' or sw_ == 'fin' or sw_ == 'finnish'):
                    name = 'Stopwords-Finnish.txt'
                elif (sw_ == 'fr' or sw_ == 'fre' or sw_ == 'fra' or sw_ == 'french'):
                    name = 'Stopwords-French.txt'
                elif (sw_ == 'de' or sw_ == 'ger' or sw_ == 'deu' or sw_ == 'german'):
                    name = 'Stopwords-German.txt'
                elif (sw_ == 'el' or sw_ == 'gre' or sw_ == 'greek'):
                    name = 'Stopwords-Greek.txt'
                elif (sw_ == 'he' or sw_ == 'heb' or sw_ == 'hebrew'):
                    name = 'Stopwords-Hebrew.txt'
                elif (sw_ == 'hi' or sw_ == 'hin' or sw_ == 'hind'):
                    name = 'Stopwords-Hind.txt'
                elif (sw_ == 'hu' or sw_ == 'hun' or sw_ == 'hungarian'):
                    name = 'Stopwords-Hungarian.txt'
                elif (sw_ == 'it' or sw_ == 'ita' or sw_ == 'italian'):
                    name = 'Stopwords-Italian.txt'
                elif (sw_ == 'ja' or sw_ == 'jpn' or sw_ == 'japanese'):
                    name = 'Stopwords-Japanese.txt'
                elif (sw_ == 'ko' or sw_ == 'kor' or sw_ == 'korean'):
                    name = 'Stopwords-Korean.txt'
                elif (sw_ == 'mr' or sw_ == 'mar' or sw_ == 'marathi'):
                    name = 'Stopwords-Marathi.txt'
                elif (sw_ == 'fa' or sw_ == 'per' or sw_ == 'fas' or sw_ == 'persian'):
                    name = 'Stopwords-Persian.txt'
                elif (sw_ == 'pl' or sw_ == 'pol' or sw_ == 'polish'):
                    name = 'Stopwords-Polish.txt'
                elif (sw_ == 'pt-br' or sw_ == 'por-br' or sw_ == 'portuguese-br'):
                    name = 'Stopwords-Portuguese-br.txt'
                elif (sw_ == 'ro' or sw_ == 'rum' or sw_ == 'ron' or sw_ == 'romanian'):
                    name = 'Stopwords-Romanian.txt'
                elif (sw_ == 'ru' or sw_ == 'rus' or sw_ == 'russian'):
                    name = 'Stopwords-Russian.txt'
                elif (sw_ == 'sk' or sw_ == 'slo' or sw_ == 'slovak'):
                    name = 'Stopwords-Slovak.txt'
                elif (sw_ == 'es' or sw_ == 'spa' or sw_ == 'spanish'):
                    name = 'Stopwords-Spanish.txt'
                elif (sw_ == 'sv' or sw_ == 'swe' or sw_ == 'swedish'):
                    name = 'Stopwords-Swedish.txt'
                elif (sw_ == 'th' or sw_ == 'tha' or sw_ == 'thai'):
                    name = 'Stopwords-Thai.txt'
                elif (sw_ == 'uk' or sw_ == 'ukr' or sw_ == 'ukrainian'):
                    name = 'Stopwords-Ukrainian.txt'
                with pkg_resources.open_binary(stws, name) as file:
                    raw_data = file.read()
                result   = chardet.detect(raw_data)
                encoding = result['encoding']
                with pkg_resources.open_text(stws, name, encoding = encoding) as file:
                    content = file.read().split('\n')
                content = [line.rstrip('\r').rstrip('\n') for line in content]
                sw      = list(filter(None, content))
                sw_full.extend(sw)
            if (verbose == True):
                print('Removing Stopwords: Working...')
            for i in range(0, len(corpus)):
               text      = corpus[i].split()
               text      = [x.replace(' ', '') for x in text if x.replace(' ', '') not in sw_full]
               corpus[i] = ' '.join(text) 
               if (verbose == True):
                   print('Removing Stopwords: ' + str(i + 1) +  ' of ' + str(len(corpus)) )
            if (verbose == True):
                print('Removing Stopwords: Done!')
        if (len(rmv_custom_words) > 0):
            if (verbose == True):
                print('Removing Custom Words: Working...')
            for i in range(0, len(corpus)):
               text      = corpus[i].split()
               text      = [x.replace(' ', '') for x in text if x.replace(' ', '') not in rmv_custom_words]
               corpus[i] = ' '.join(text) 
               if (verbose == True):
                   print('Removing Custom Words: ' + str(i + 1) +  ' of ' + str(len(corpus)) )
            if (verbose == True):
                print('Removing Custom Word: Done!')
        if (rmv_accents == True):
            if (verbose == True):
                print('Removing Accents: Working...')
            for i in range(0, len(corpus)):
                text = corpus[i]
                try:
                    text = unicode(text, 'utf-8')
                except NameError: 
                    pass
                text      = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
                corpus[i] = str(text)
            if (verbose == True):
                print('Removing Accents: Done!')
        # Remove Numbers
        if (rmv_numbers == True):
            if (verbose == True):
                print('Removing Numbers: Working...')
            corpus = [re.sub('[0-9]', ' ', i) for i in corpus] 
            if (verbose == True):
                print('Removing Numbers: Done!')
        for i in range(0, len(corpus)):
            corpus[i] = ' '.join(corpus[i].split())
        return corpus

    # Function: TF-IDF
    def dtm_tf_idf(self, corpus):
        vectorizer = TfidfVectorizer(norm = 'l2')
        tf_idf     = vectorizer.fit_transform(corpus)
        try:
            tokens = vectorizer.get_feature_names_out()
        except:
            tokens = vectorizer.get_feature_names()
        values     = tf_idf.todense()
        values     = values.tolist()
        dtm        = pd.DataFrame(values, columns = tokens)
        return dtm
   
    # Function: Projection
    def docs_projection(self, view = 'browser', corpus_type = 'abs', stop_words = ['en'], rmv_custom_words = [], custom_label = [], custom_projection = [], n_components = 2, n_clusters = 5, node_labels = True, node_size = 25, node_font_size = 10, tf_idf = True, embeddings = False, method = 'tsvd', model = 'allenai/scibert_scivocab_uncased', showlegend = False, cluster_method = 'kmeans', min_size = 5, max_size = 15):
        if   (corpus_type == 'abs'):
            corpus = self.data['abstract']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'title'):
            corpus = self.data['title']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'kwa'): 
            corpus = self.data['author_keywords']
            corpus = corpus.tolist()
        elif (corpus_type == 'kwp'):
            corpus = self.data['keywords']
            corpus = corpus.tolist()
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (embeddings == True):
            model = SentenceTransformer(model) # 'allenai/scibert_scivocab_uncased'; 'all-MiniLM-L6-v2'
            embds = model.encode(corpus)
        dtm = self.dtm_tf_idf(corpus)
        if (method.lower() == 'umap'):
            decomposition = UMAP(n_components = n_components, random_state = 1001)
        else:
            decomposition = tsvd(n_components = n_components, random_state = 1001)
        if (len(custom_projection) == 0 and embeddings == False):
            transformed = decomposition.fit_transform(dtm)
        elif (len(custom_projection) == 0 and embeddings == True):
            transformed = decomposition.fit_transform(embds)
        elif (custom_projection.shape[0] == self.data.shape[0] and custom_projection.shape[1] >= 2):
            transformed = np.copy(custom_projection)
        if (len(custom_label) == 0):
            if cluster_method == 'kmeans':
                cluster = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 100, max_iter = 10, random_state = 1001)
                if (tf_idf == True and embeddings == False):
                    cluster.fit(dtm)
                else:
                    cluster.fit(transformed)
                labels  = cluster.labels_
                n       = len(set(labels.tolist()))
            else:
                cluster = HDBSCAN(min_cluster_size = min_size, max_cluster_size = max_size)
                if (tf_idf == True and embeddings == False):
                    cluster.fit(dtm)
                else:
                    cluster.fit(transformed)
                labels  = cluster.labels_
                n       = len(set(labels.tolist())) 
        elif (len(custom_label) > 0):
            labels = [item for item in custom_label]
            n      = len(set(labels))
        n_trace = []
        for i in range(0, n):
            labels_c    = []
            node_list   = []
            n_id        = []
            idx         = [j for j in range(0, len(labels)) if labels[j] == i]
            x           = transformed[idx, 0]
            y           = transformed[idx, 1]
            labels_c.extend(self.color_names[i] for item in idx)
            if (node_labels == True):
                node_list.extend(idx)
            else:
                idx_ = ['' for item in idx]
                node_list.extend(idx_)
            for j in range(0, len(idx)):
                n_id.append(
                            'id:' +str(idx[j])               +'<br>'  +
                            'cluster:' +str(i)               +'<br>'  +
                             self.data.loc[idx[j], 'author'] +' ('    +
                             self.data.loc[idx[j], 'year']   +'). '   +
                             self.data.loc[idx[j], 'title']  +'. '    +
                             self.data.loc[idx[j], 'journal']+'. doi:'+
                             self.data.loc[idx[j], 'doi']    +'.'
                             )
                n_id[-1] = '<br>'.join(textwrap.wrap(n_id[-1], width = 50))
            n_trace.append(go.Scatter(x         = x,
                                      y         = y,
                                      opacity   = 1,
                                      mode      = 'markers+text',
                                      marker    = dict(symbol = 'circle-dot', size = node_size, color = self.color_names[i]),
                                      text      = node_list,
                                      hoverinfo = 'text',
                                      hovertext = n_id,
                                      name      = f'Cluster {i}'
                                      ))
            
        layout  = go.Layout(showlegend   = showlegend,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = '#f5f5f5',
                            xaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'white',
                                                  zeroline       = False, 
                                                  showticklabels = False, 
                                               ),
                            yaxis        = dict(  showgrid       = True,  
                                                  gridcolor      = 'white',
                                                  zeroline       = False, 
                                                  showticklabels = False,
                                                )
                            )
        fig_proj = go.Figure(data = n_trace, layout = layout)
        fig_proj.update_traces(textfont_size = node_font_size, textfont_color = 'white') 
        fig_proj.show() 
        return transformed, labels

    #############################################################################
    
    # Function: References Citations 
    def ref_citation_matrix(self, tgt_ref_id = [], date_start = None, date_end = None):
        citation_records = []
        for article_idx, pub_year in enumerate(self.dy):
            year = int(pub_year) 
            if (date_start is not None and year < date_start) or (date_end is not None and year > date_end):
                continue
            ref_names = self.ref[article_idx]
            ref_ids   = self.ref_id[article_idx]
            for ref_name, ref_id in zip(ref_names, ref_ids):
                if (ref_name.lower() == 'unknown'):
                    continue
                if tgt_ref_id and ref_id not in tgt_ref_id:
                    continue
                citation_records.append({'Reference': ref_name, 'Reference ID': ref_id, 'Citing Articles': (article_idx, year)})
        df                          = pd.DataFrame(citation_records)
        result_df                   = df.groupby('Reference').agg({'Reference ID': lambda x: x.iloc[0], 'Citing Articles': lambda x: list(set(x))}).reset_index()
        ref_year_dict = dict(zip(self.u_ref, self.dy_ref))
        result_df['Reference Year'] = result_df['Reference'].map(ref_year_dict)
        return result_df[['Reference', 'Reference ID', 'Reference Year', 'Citing Articles']]
 
    # Function: Top References
    def plot_top_refs(self, view = 'browser', topn = 10, font_size = 8, use_ref_id = False, date_start = None, date_end = None):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        all_refs_names = [r for refs in self.ref    for r in refs if r != 'UNKNOWN']
        all_refs_ids   = [r for refs in self.ref_id for r in refs if r != 'UNKNOWN']
        ref_year_names = {ref: year for ref, year in zip(self.u_ref,    self.dy_ref)}
        ref_year_ids   = {ref: year for ref, year in zip(self.u_ref_id, self.dy_ref)}
        if (use_ref_id == False):
            ref_year_dict = ref_year_names
            all_refs      = all_refs_names
        else:
            ref_year_dict = ref_year_ids
            all_refs      = all_refs_ids
        if (date_start is not None or date_end is not None):
            filtered_all_refs       = []
            filtered_all_refs_names = []
            filtered_all_refs_ids   = []
            for i in range(0, len(all_refs)):
                year = ref_year_dict.get(all_refs_ids[i], -1)
                if (year == -1):
                    continue
                if (date_start is not None and year < date_start):
                    continue
                if (date_end is not None and year > date_end):
                    continue
                filtered_all_refs.append(all_refs[i])
                filtered_all_refs_names.append(all_refs_names[i])
                filtered_all_refs_ids.append(all_refs_ids[i])
        else:
            filtered_all_refs       = all_refs
            filtered_all_refs_names = all_refs_names
            filtered_all_refs_ids   = all_refs_ids
        ref_counter       = Counter(filtered_all_refs)
        top_refs          = ref_counter.most_common(topn)
        ref_counter_names = Counter(filtered_all_refs_names)
        top_refs_names    = ref_counter_names.most_common(topn)
        ref_counter_ids   = Counter(filtered_all_refs_ids)
        top_refs_ids      = ref_counter_ids.most_common(topn)
        if (top_refs):
            labels, values = zip(*top_refs)
            labels_n, _    = zip(*top_refs_names)
            labels_i, _    = zip(*top_refs_ids)
        else:
            labels, values = [], []
            labels_n       = []
            labels_i       = []
        years_top_refs = [ref_year_dict.get(ref, -1) for ref in labels]
        self.top_refs  = pd.DataFrame({'Reference': labels_n, 'Reference ID': labels_i, 'Reference Year': years_top_refs, 'Raw Citation Counts': values})
        fig            = go.Figure(data = [go.Pie(
                                                    labels       = labels,
                                                    values       = values,
                                                    textinfo     = 'value',        
                                                    textposition = 'inside',
                                                    hoverinfo    = 'label+value+percent',
                                                    marker       = dict(colors = self.color_names, line = dict(color = 'black', width = 1) )
                                                )])
        fig.update_traces(domain = {'x': [0, 1], 'y': [0.3, 1]})
        fig.update_layout(
                            title_text = 'Top Cited References',
                            legend     = dict(
                                                orientation = 'h',      
                                                yanchor     = 'top',
                                                y           = 0.28,              
                                                xanchor     = 'center',
                                                x           = 0.5,
                                                font        = dict(size = font_size)), 
                                                margin      = dict(l = 50, r = 50, t = 50, b = 50)
                                               )
        fig.show()
        return
    
    # Function: Citation Trajectory
    def plot_citation_trajectory(self, view = 'browser', ref_names = [], ref_ids = []):
        if( view == 'browser'):
            pio.renderers.default = 'browser'
        if (ref_names):
            selected_refs = ref_names
            use_names     = True
        elif (ref_ids):
            selected_refs = ref_ids
            use_names     = False   
        valid_years         = [int(year) for year in self.dy if year != -1]
        min_year, max_year  = min(valid_years), max(valid_years)
        x_range             = list(range(min_year, max_year + 1))
        citation_trajectory = {ref: {year: 0 for year in x_range} for ref in selected_refs}
        for i, pub_year in enumerate(self.dy):
            if (pub_year == -1):
                continue
            article_refs = self.ref[i] if use_names else self.ref_id[i]
            c            = 0
            for r in article_refs:
                c = c + 1
                if (r in selected_refs):
                    if (pub_year in citation_trajectory[r]):
                        citation_trajectory[r][pub_year] =  citation_trajectory[r][pub_year] + 1
        fig = go.Figure()
        for idx, ref in enumerate(selected_refs):
            x_values = [year for year in x_range if citation_trajectory[ref][year] != 0]
            y_values = [citation_trajectory[ref][year] for year in x_range if citation_trajectory[ref][year] != 0]
            c        = self.color_names[idx % len(self.color_names)]
            fig.add_trace(go.Scatter(
                                        x             = x_values,
                                        y             = y_values,
                                        mode          = 'lines+markers',
                                        name          = ref,
                                        line          = dict(color = c, width = 2.5, shape  = 'spline'),
                                        marker        = dict(color = c, size = 8,    symbol = 'circle'),
                                        text          = [f'Citations: {count}' for count in y_values],
                                        textposition  = 'top center',
                                        hoverinfo     = 'x+y+name+text',
                                        hovertemplate='<b>Year: %{x}<br>Citations: %{y}<extra></extra>'
                                    ))
        fig.update_layout(
                            title  = dict( text = 'Citation Trajectory Analysis', font = dict(color = '#2a2a2a'), x = 0.5, xanchor = 'center'),
                            xaxis  = dict( title = 'Publication Year', showgrid = True, gridcolor = 'white', tickmode = 'linear', dtick = 2),
                            yaxis  = dict(title = 'Citation Count', rangemode = 'tozero', gridcolor = 'white'),
                            legend = dict(title = dict(text = 'References'), orientation = 'v', yanchor = 'top', y = 1, xanchor = 'left', x = 1.02, bgcolor = 'rgba(255,255,255,0.5)'),
                            margin = dict(l = 50, r = 50, t = 50, b = 50)
                          )
        fig.show()
        return
    
    # Function: RPYS (Reference Publication Year Spectroscopy) with Gaussian Filter to Find Peaks
    def plot_rpys(self, view = 'browser', peaks_only = False):
        if( view == 'browser'):
            pio.renderers.default = 'browser'
        publication_years = [item for item in self.dy_ref if item != -1]
        year_counts       = Counter(publication_years)
        years             = sorted(year_counts.keys())
        counts            = np.array([year_counts[year] for year in years])
        smoothed_counts   = gaussian_filter1d(counts, sigma = 1)
        peaks, properties = find_peaks(smoothed_counts, height = 1)
        peak_years        = np.array(years)[peaks]
        peak_values       = smoothed_counts[peaks]
        self.rpys_pk      = pd.DataFrame({'Peak Years': peak_years, 'Counts': peak_values})
        self.rpys_rs      = pd.DataFrame({'Years': years, 'Raw Citation Counts': counts, 'Smoothed Citation Counts': smoothed_counts})
        bar_colors        = ['rgba(240, 100, 100, 0.5)' if year in peak_years else 'rgba(100, 150, 240, 0.5)' for year in years]
        if (peaks_only == True):
            for i in range(len(bar_colors)-1, -1, -1):
                if (bar_colors[i] != 'rgba(240, 100, 100, 0.5)' ):
                    del years[i]
                    del bar_colors[i]
                    counts          = np.delete(counts, i)  
                    smoothed_counts = np.delete(smoothed_counts, i) 
        fig = go.Figure()
        fig.add_trace(go.Bar(
                                    x            = years,
                                    y            = counts,
                                    name         = 'Raw Citation Counts',
                                    marker_color = bar_colors
                            ))
        fig.add_trace(go.Scatter(
                                    x            = years,
                                    y            = smoothed_counts,
                                    mode         = 'lines+markers',
                                    name         = 'Smoothed Citation Counts',
                                    line         = dict(color = 'black', width = 1)
                            ))
        fig.add_trace(go.Scatter(
                                    x            = peak_years,
                                    y            = peak_values,
                                    mode         = 'markers',
                                    marker       = dict(color = 'red', size = 10, symbol = 'circle'),
                                    name         = 'Peaks'
                            ))
        fig.update_layout(
                                    title        = 'Reference Publication Year Spectroscopy (RPYS)',
                                    xaxis_title  = 'Publication Year',
                                    yaxis_title  = 'Citation Counts',
                                    showlegend   = True,
                                    xaxis        = dict(rangeselector = dict( buttons = list([ dict(count = 1, label = '1y', step = 'year', stepmode = 'backward'), dict(step = 'all') ])), 
                                                        rangeslider  = dict(visible = True), type = 'date')
                            )
        fig.show()
        return

    # Function: Top Cited Co-References
    def top_cited_co_references(self, group = 2, topn = 10):
        co_cited_groups = []
        for refs in self.ref_id:
            refs = list(set(refs))
            for combo in combinations(refs, group):
                co_cited_groups.append(tuple(sorted(combo)))
        group_counts = Counter(co_cited_groups)
        top_groups   = group_counts.most_common(topn)
        df           = pd.DataFrame(top_groups, columns = ['Reference ID Sets', 'Count'])
        return df

    # Function: Plot Co-Citation
    def plot_co_citation_network(self, view = 'browser', target_ref_id = '', topn = 10):
        if( view == 'browser'):
            pio.renderers.default = 'browser'
        citing_articles    = [idx for idx, refs in enumerate(self.ref_id) if target_ref_id in refs]
        co_cited_refs      = []
        for article_idx in citing_articles:
            co_cited_refs.extend(self.ref_id[article_idx])
        co_cited_refs      = [ref for ref in co_cited_refs if ref != target_ref_id]
        co_cited_counts    = Counter(co_cited_refs)
        top_co_cited       = co_cited_counts.most_common(topn)
        ref_details        = []
        edges              = []
        counts             = [count for _, count in top_co_cited]
        max_count          = max(counts) if counts else 1
        min_size, max_size = 10, 20
        for ref_id, count in top_co_cited:
            if (ref_id in self.u_ref_id):
                idx      = self.u_ref_id.index(ref_id)
                ref_name = self.u_ref[idx]
                ref_year = self.dy_ref[idx]
                ref_details.append((ref_id, ref_name, ref_year, count))
                edges.append((target_ref_id, ref_id, count))
        self.top_co_c = pd.DataFrame(ref_details, columns = ['Reference ID', 'Reference', 'Year', 'Count'])
        G             = nx.Graph()
        G.add_node(target_ref_id, 
                   label     = target_ref_id, 
                   size      = 40, 
                   color     = '#FF6B6B', 
                   hovertext = f"Target: {target_ref_id}") 
        for ref_id, ref_name, ref_year, count in ref_details:
            title         = f"{ref_name} ({ref_year})"
            wrapped_title = '<br>'.join(textwrap.wrap(title, width=75))
            hover_text    = f"{wrapped_title}<br>Co-Citations: {count}"
            norm_size     = (count / max_count) * (max_size - min_size) + min_size
            G.add_node(ref_id, 
                       label     = ref_id, 
                       size      = norm_size, 
                       color     = '#657beb', 
                       hovertext = hover_text)
        for source, target, weight in edges:
            G.add_edge(source, target, weight = weight)
        pos = nx.spring_layout(G, seed = 42, k = 0.6, iterations = 100)
        edge_x, edge_y, edge_weights = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        edge_trace = go.Scatter(
                                x         = edge_x, 
                                y         = edge_y,
                                line      = dict(width = 0.15, color = 'black'),
                                hoverinfo = 'none',
                                mode      = 'lines'
                            )
        node_x, node_y, node_text, node_size, node_color, node_hovertext = [], [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_size.append(G.nodes[node]['size'])
            node_color.append(G.nodes[node]['color'])
            node_hovertext.append(G.nodes[node]['hovertext'])
        node_trace = go.Scatter(
                                x            = node_x, 
                                y            = node_y,
                                mode         = 'markers+text',
                                text         = node_text,
                                marker       = dict(size = node_size, color = node_color, line = dict(width = 1.5, color = 'black')),
                                textposition = 'top center',
                                hoverinfo    = 'text',
                                hovertext    = node_hovertext,
                                textfont     = dict(size = 10, color = '#2D3436')
                                )
        fig = go.Figure(data = [edge_trace, node_trace])
        fig.update_layout(
                            showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 0, l = 0, r = 0, t = 40),
                            plot_bgcolor = 'rgb(255, 255, 256)',
                            xaxis        = dict(showgrid = False, zeroline = False, showticklabels = False, scaleanchor = 'y'),
                            yaxis        = dict(showgrid = False, zeroline = False, showticklabels = False, scaleanchor = 'x')
                        )
        fig.show()
        return 
 
    # Function: Salsa (Stochastic Approximation for Local Search in Assignment Problems)
    def salsa(self, max_iter = 150, tol = 1e-6, topn_decade = 5):
        
        #----------------------------------------------------------------------
        
        def build_hubs_authorities(A, hubs, authorities, out_degrees, in_degrees, max_iter, tol):
            for it in range(0, max_iter):
                old_hubs = hubs.copy()
                old_auth = authorities.copy()
                factor1  = np.divide(hubs, out_degrees, out = np.zeros_like(hubs), where = (out_degrees != 0))
                new_auth = np.dot(A.T, factor1)
                factor2  = np.divide(new_auth, in_degrees, out = np.zeros_like(new_auth), where = (in_degrees != 0))
                new_hub  = np.dot(A, factor2)
                sum_auth = new_auth.sum()
                if (sum_auth > 0):
                    new_auth = new_auth / sum_auth
                sum_hub = new_hub.sum()
                if (sum_hub > 0):
                    new_hub = new_hub / sum_hub
                hubs = new_hub
                authorities = new_auth
                if (np.abs(hubs - old_hubs).sum() < tol and np.abs(authorities - old_auth).sum() < tol):
                    break
            return hubs, authorities
     
        #----------------------------------------------------------------------
        
        node_mapping = {}  
        node_ids     = []  
        node_years   = []   
        for i in range(0, len(self.ref)):
            key = str(i)
            if key not in node_mapping:
                node_mapping[key] = len(node_ids)
                node_ids.append(key)
                node_years.append(int(self.dy[i]) if i < len(self.dy) else -1)
        u_ref_dict = {}
        for i, ref_str in enumerate(self.u_ref):
            identifier          = self.u_ref_id[i]
            key                 = str(identifier)
            year                = self.dy_ref[i]
            u_ref_dict[ref_str] = (key, year)
            if key not in node_mapping:
                node_mapping[key] = len(node_ids)
                node_ids.append(key)
                node_years.append(year)
            else:
                idx = node_mapping[key]
                if (node_years[idx] == -1 and year != -1):
                    node_years[idx] = year
        N = len(node_ids)
        A = np.zeros((N, N))
        for i, citations in enumerate(self.ref):
            source_key = str(i)
            source_idx = node_mapping[source_key]
            for citation in citations:
                if (citation in u_ref_dict):
                    target_key, _             = u_ref_dict[citation]
                    target_idx                = node_mapping[target_key]
                    A[source_idx, target_idx] = 1  
        hubs              = np.ones(N) / N
        authorities       = np.ones(N) / N
        out_degrees       = A.sum(axis = 1)  
        in_degrees        = A.sum(axis = 0)   
        hubs, authorities = build_hubs_authorities(A, hubs, authorities, out_degrees, in_degrees, max_iter, tol)
        year_aggregates   = {}
        for idx in range(0, N):
            year = node_years[idx]
            if (year == -1):
                continue
            if year not in year_aggregates:
                year_aggregates[year] = {'hub_scores': [], 'authority_scores': []}
            year_aggregates[year]['hub_scores'].append(hubs[idx])
            year_aggregates[year]['authority_scores'].append(authorities[idx])
        for year in year_aggregates:
            hs                    = year_aggregates[year]['hub_scores']
            as_                   = year_aggregates[year]['authority_scores']
            count                 = len(hs)
            year_aggregates[year] = {'mean_hub': np.mean(hs), 'mean_authority': np.mean(as_), 'count': count}
        result       = {'hubs': hubs, 'authorities': authorities, 'node_ids': node_ids, 'node_years': node_years, 'year_aggregates': year_aggregates}
        decade_stats = {}
        for idx, year in enumerate(result['node_years']):
            if (year == -1):
                continue  
            decade = int(year // 10 * 10)
            if decade not in decade_stats:
                decade_stats[decade] = {'authorities': [], 'hubs': []}
            decade_stats[decade]['authorities'].append(result['authorities'][idx])
            decade_stats[decade]['hubs'].append(result['hubs'][idx])
        decades          = sorted(decade_stats.keys())
        top_by_decade_a  = {}
        top_by_decade_h  = {}
        topn             = topn_decade
        for decade in decades:
            indices        = [idx for idx, year in enumerate(result['node_years']) if year != -1 and int(year // 10 * 10) == decade]
            sorted_indices = sorted(indices, key = lambda i: result['authorities'][i], reverse = True)
            top_by_decade_a[decade] = [(result['node_ids'][i], result['authorities'][i]) for i in sorted_indices[:topn]]
            sorted_indices = sorted(indices, key = lambda i: result['hubs'][i], reverse = True)
            top_by_decade_h[decade] = [(result['node_ids'][i], result['hubs'][i]) for i in sorted_indices[:topn]]
        return result, top_by_decade_a, top_by_decade_h

    # Function: Detect Sleeping Beauties. Based on < https://doi.org/10.1007/s41109-021-00389-0 >
    def detect_sleeping_beauties(self, topn = 10, min_count = 10): 
        valid_years         = [int(year) for year in self.dy if year != -1]
        min_year, max_year  = min(valid_years), max(valid_years)
        x_range             = list(range(min_year, max_year + 1))
        citation_trajectory = {ref: {year: 0 for year in x_range} for ref in self.u_ref_id}
        for i, pub_year in enumerate(self.dy):
            if (pub_year == -1):
                continue
            article_refs = self.ref_id[i]
            c            = 0
            for r in article_refs:
                c = c + 1
                if (r in self.u_ref_id):
                    if (pub_year in citation_trajectory[r]):
                        citation_trajectory[r][pub_year] =  citation_trajectory[r][pub_year] + 1
        metrics = {}
        for ref, counts_dict in citation_trajectory.items():
            years = sorted(counts_dict.keys())
            if not years:
                continue
            pub_year        = years[0]
            t_values        = [year - pub_year   for year in years]
            citations       = [counts_dict[year] for year in years]
            total_citations = sum(citations)
            if (total_citations < min_count):
                metrics[ref] = {'B': 0.0, 't_a': 0, 'c0': citations[0], 'cm': max(citations), 't_m': 0}
                continue
            c0        = citations[0]
            cm        = max(citations)
            t_m_index = citations.index(cm)
            t_m       = t_values[t_m_index]
            if (t_m == 0):
                metrics[ref] = {'B': 0.0, 't_a': 0, 'c0': c0, 'cm': cm, 't_m': 0}
                continue
            L = lambda t: ((cm - c0) / t_m) * t + c0
            B = 0.0
            for t, c in zip(t_values, citations):
                if (t > t_m):
                    break
                B = B + (L(t) - c) / max(1, c)
            denom    = np.sqrt((cm - c0)**2 + t_m**2)
            d_values = []
            for t, c in zip(t_values, citations):
                if (t > t_m):
                    break
                d = abs((cm - c0)*t - t_m*c + t_m*c0) / np.maximum(1, denom) 
                d_values.append(d)
            if (d_values):
                max_idx = np.argmax(d_values)
                t_a     = t_values[max_idx]
            else:
                t_a     = 0
            metrics[ref] = {'B': B, 'SBI': B / denom if denom > 0 else 0.0, 't_a': t_a, 'c0': c0, 'cm': cm, 't_m': t_m}
        metrics = sorted(metrics.items(), key = lambda item: item[1]['B'], reverse = True)
        metrics = [item for item in metrics if item[1]['B'] > 0][:topn]
        metrics = pd.DataFrame([{ 'ref': ref, **vals } for ref, vals in metrics])
        return metrics

    # Function: Detect Princes. Based on < https://doi.org/10.1007/s41109-021-00389-0 >
    def detect_princes(self, metrics):
        results = []
        for _, row in metrics.iterrows():
            sb_id          = row['ref']
            citing_indices = [idx for idx, refs in enumerate(self.ref_id) if sb_id in refs]
            sb_citers      = []
            for idx in citing_indices:
                pub_year = self.dy[idx]
                if (pub_year == -1):
                    continue
                sb_citers.append({'id': self.table_id_doc['ID'][idx], 'pub_year': int(pub_year), 'references': self.ref_id[idx]})
            candidates         = [ p for p in sb_citers ] 
            co_citation_counts = {}
            for candidate in candidates:
                cid                     = candidate['id']
                count                   = sum(1 for p in sb_citers if cid in p.get('references', []))
                co_citation_counts[cid] = count
            if (co_citation_counts):
                prince_id = max(co_citation_counts, key=co_citation_counts.get)
                prince    = next(p for p in candidates if p['id'] == prince_id)
                results.append({ 'SB_id': sb_id, 'PR_id': prince_id, 'PR_pub_year': prince['pub_year'], 'co_citation_count': co_citation_counts[prince_id]})
            else:
                results.append({'SB_id': sb_id, 'PR_id': None, 'PR_pub_year': None, 'co_citation_count': 0})
        results_df = pd.DataFrame(results)
        return results_df

    #############################################################################
        
    # Function: Authors Colaboration Adjacency Matrix   
    def __adjacency_matrix_aut(self, min_colab = 1):
        tgt_entry      = self.aut
        tgt_entry_u    = self.u_aut
        tgt_label      = 'a_'
        item_to_idx    = {tgt: idx for idx, tgt in enumerate(tgt_entry_u)} 
        n_items        = len(tgt_entry_u) 
        item_idx_list  = []
        for it in tgt_entry:
            item_idx       = np.array([item_to_idx[i] for i in it], dtype = np.int32)
            item_idx_list.append(item_idx)
        item_idx_list_typed = List()
        for arr in item_idx_list:
            item_idx_list_typed.append(arr)
        rows, cols     = build_edges(item_idx_list_typed)
        data           = np.ones(len(rows), dtype = np.int8)
        adjacency      = coo_matrix((data, (rows, cols)), shape = (n_items, n_items)).tocsr()
        n_colab        = np.array(adjacency.sum(axis = 0)).flatten()
        adjacency.data = (adjacency.data > 0).astype(np.int8)
        if (min_colab > 0):
            low_colab_mask = n_colab < min_colab
            if (np.any(low_colab_mask)):
                low_idx               = np.where(low_colab_mask)[0]
                adjacency[low_idx, :] = 0
                adjacency[:, low_idx] = 0
        adjacency.eliminate_zeros()
        self.matrix_a = pd.DataFrame.sparse.from_spmatrix(adjacency, index = tgt_entry_u, columns = tgt_entry_u)
        self.labels_a = [tgt_label + str(i) for i in range(0, n_items)]
        self.n_colab  = n_colab.tolist()
        return
    
    # Function: Country Colaboration Adjacency Matrix
    def __adjacency_matrix_ctr(self, min_colab = 1):
        tgt_entry      = self.ctr
        tgt_entry_u    = self.u_ctr
        tgt_label      = 'c_'
        item_to_idx    = {tgt: idx for idx, tgt in enumerate(tgt_entry_u)} 
        n_items        = len(tgt_entry_u) 
        item_idx_list  = []
        for it in tgt_entry:
            item_idx       = np.array([item_to_idx[i] for i in it], dtype = np.int32)
            item_idx_list.append(item_idx)
        item_idx_list_typed = List()
        for arr in item_idx_list:
            item_idx_list_typed.append(arr)
        rows, cols     = build_edges(item_idx_list_typed)
        data           = np.ones(len(rows), dtype = np.int8)
        adjacency      = coo_matrix((data, (rows, cols)), shape = (n_items, n_items)).tocsr()
        n_colab        = np.array(adjacency.sum(axis = 0)).flatten()
        adjacency.data = (adjacency.data > 0).astype(np.int8)
        if (min_colab > 0):
            low_colab_mask = n_colab < min_colab
            if (np.any(low_colab_mask)):
                low_idx               = np.where(low_colab_mask)[0]
                adjacency[low_idx, :] = 0
                adjacency[:, low_idx] = 0
        adjacency.eliminate_zeros()
        self.matrix_a = pd.DataFrame.sparse.from_spmatrix(adjacency, index = tgt_entry_u, columns = tgt_entry_u)
        self.labels_a = [tgt_label + str(i) for i in range(0, n_items)]
        self.n_colab  = n_colab.tolist()
        return

    # Function: Institution Colaboration Adjacency Matrix   
    def __adjacency_matrix_inst(self, min_colab = 1):
        tgt_entry      = self.uni
        tgt_entry_u    = self.u_uni
        tgt_label      = 'i_'
        item_to_idx    = {tgt: idx for idx, tgt in enumerate(tgt_entry_u)} 
        n_items        = len(tgt_entry_u) 
        item_idx_list  = []
        for it in tgt_entry:
            item_idx       = np.array([item_to_idx[i] for i in it], dtype = np.int32)
            item_idx_list.append(item_idx)
        item_idx_list_typed = List()
        for arr in item_idx_list:
            item_idx_list_typed.append(arr)
        rows, cols     = build_edges(item_idx_list_typed)
        data           = np.ones(len(rows), dtype = np.int8)
        adjacency      = coo_matrix((data, (rows, cols)), shape = (n_items, n_items)).tocsr()
        n_colab        = np.array(adjacency.sum(axis = 0)).flatten()
        adjacency.data = (adjacency.data > 0).astype(np.int8)
        if (min_colab > 0):
            low_colab_mask = n_colab < min_colab
            if (np.any(low_colab_mask)):
                low_idx               = np.where(low_colab_mask)[0]
                adjacency[low_idx, :] = 0
                adjacency[:, low_idx] = 0
        adjacency.eliminate_zeros()
        self.matrix_a = pd.DataFrame.sparse.from_spmatrix(adjacency, index = tgt_entry_u, columns = tgt_entry_u)
        self.labels_a = [tgt_label + str(i) for i in range(0, n_items)]
        self.n_colab  = n_colab.tolist()
        return
    
    # Function: KWA Colaboration Adjacency Matrix   
    def __adjacency_matrix_kwa(self, min_colab = 1):
        tgt_entry      = self.auk
        tgt_entry_u    = self.u_auk
        item_to_idx    = {tgt: idx for idx, tgt in enumerate(tgt_entry_u)} 
        n_items        = len(tgt_entry_u) 
        item_idx_list  = []
        for it in tgt_entry:
            if (it[0] != 'unknown'):
                item_idx       = np.array([item_to_idx[i] for i in it], dtype = np.int32)
                item_idx_list.append(item_idx)
        item_idx_list_typed = List()
        for arr in item_idx_list:
            item_idx_list_typed.append(arr)
        rows, cols     = build_edges(item_idx_list_typed)
        data           = np.ones(len(rows), dtype = np.int8)
        adjacency      = coo_matrix((data, (rows, cols)), shape = (n_items, n_items)).tocsr()
        n_colab        = np.array(adjacency.sum(axis = 0)).flatten()
        adjacency.data = (adjacency.data > 0).astype(np.int8)
        if (min_colab > 0):
            low_colab_mask = n_colab < min_colab
            if (np.any(low_colab_mask)):
                low_idx               = np.where(low_colab_mask)[0]
                adjacency[low_idx, :] = 0
                adjacency[:, low_idx] = 0
        adjacency.eliminate_zeros()
        self.matrix_a = pd.DataFrame.sparse.from_spmatrix(adjacency, index = tgt_entry_u, columns = tgt_entry_u)
        self.labels_a = [self.dict_kwa_id[item] for item in tgt_entry_u] 
        self.n_colab  = n_colab.tolist()
        return
    
    # Function: KWP Colaboration Adjacency Matrix
    def __adjacency_matrix_kwp(self, min_colab = 1):
        tgt_entry      = self.kid
        tgt_entry_u    = self.u_kid
        item_to_idx    = {tgt: idx for idx, tgt in enumerate(tgt_entry_u)} 
        n_items        = len(tgt_entry_u) 
        item_idx_list  = []
        for it in tgt_entry:
            if (it[0] != 'unknown'):
                item_idx       = np.array([item_to_idx[i] for i in it], dtype = np.int32)
                item_idx_list.append(item_idx)
        item_idx_list_typed = List()
        for arr in item_idx_list:
            item_idx_list_typed.append(arr)
        rows, cols     = build_edges(item_idx_list_typed)
        data           = np.ones(len(rows), dtype = np.int8)
        adjacency      = coo_matrix((data, (rows, cols)), shape = (n_items, n_items)).tocsr()
        n_colab        = np.array(adjacency.sum(axis = 0)).flatten()
        adjacency.data = (adjacency.data > 0).astype(np.int8)
        if (min_colab > 0):
            low_colab_mask = n_colab < min_colab
            if (np.any(low_colab_mask)):
                low_idx               = np.where(low_colab_mask)[0]
                adjacency[low_idx, :] = 0
                adjacency[:, low_idx] = 0
        adjacency.eliminate_zeros()
        self.matrix_a = pd.DataFrame.sparse.from_spmatrix(adjacency, index = tgt_entry_u, columns = tgt_entry_u)
        self.labels_a = [self.dict_kwp_id[item] for item in tgt_entry_u] 
        self.n_colab  = n_colab.tolist()
        return

    # Function: References Adjacency Matrix   
    def __adjacency_matrix_ref(self, min_cites = 2, local_nodes = False):
        u_ref_map   = {ref: idx for idx, ref in enumerate(self.u_ref)}
        num_rows    = self.data.shape[0]
        num_cols    = len(self.u_ref)
        ref_indices = []
        for refs in self.ref:
            filtered = [u_ref_map[r] for r in refs if r in u_ref_map]
            ref_indices.append(np.array(filtered, dtype = np.int32))
        ref_idx_list_typed = List()
        for arr in ref_indices:
            ref_idx_list_typed.append(arr)
        row_indices, col_indices = build_edges_ref(ref_idx_list_typed)
        data                     = np.ones(len(row_indices), dtype = np.float32)
        sparse_matrix            = csr_matrix((data, (row_indices, col_indices)), shape = (num_rows, num_cols), dtype = np.float32)
        self.matrix_r            = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns = self.u_ref)
        self.labels_r            = [f'r_{i}' for i in range(0, num_cols)]
        sources                  = self.data['source'].str.lower()
        keys_1                   = self.data['title'].str.lower().str.replace('[', '', regex = False).str.replace(']', '', regex = False).tolist()
        keys_2                   = self.data['doi'].str.lower().tolist()
        keys                     = np.where(sources.isin(['scopus', 'pubmed']), keys_1, np.where(sources == 'wos', keys_2, None))
        corpus                   = ' '.join(ref.lower() for ref in self.u_ref)
        matched_indices          = []
        for i, key in enumerate(keys):
            if (key and key.strip()):
                try:
                    compiled_regex = re.compile(key)
                    if (re.search(compiled_regex, corpus)):
                        matched_indices.append(i)
                except:
                    pass
        insd_r      = []
        insd_t      = []
        u_ref_lower = [ref.lower() for ref in self.u_ref]
        for i in matched_indices:
            key = keys[i]
            for j, ref in enumerate(u_ref_lower):
                if (re.search(key, ref)):
                    insd_r.append(f'r_{j}')
                    insd_t.append(str(i))
                    self.dy_ref[j] = int(self.dy[i])
                    break
        self.dict_lbs         = dict(zip(insd_r, insd_t))
        self.dict_lbs.update({label: label for label in self.labels_r if label not in self.dict_lbs})
        self.labels_r         = [self.dict_lbs.get(label, label) for label in self.labels_r]
        self.matrix_r.columns = self.labels_r
        if (local_nodes):
            mask          = ~self.matrix_r.columns.str.contains('r_')
            self.matrix_r = self.matrix_r.loc[:, mask]
            self.labels_r = self.matrix_r.columns.tolist()
        if (min_cites >= 1):
            col_sums      = self.matrix_r.sum(axis = 0)
            cols_to_keep  = col_sums[col_sums >= min_cites].index
            self.matrix_r = self.matrix_r[cols_to_keep]
            self.labels_r = cols_to_keep.tolist()
        self.matrix_r = self.matrix_r.astype(pd.SparseDtype('float', 0))
        return

    # Function: Make Matrix
    def make_matrix(self, entry = 'aut', min_count = 0, local_nodes = False):
        if   (entry == 'aut'):
            self.__adjacency_matrix_aut(min_count)
            return self.matrix_a
        elif (entry == 'cout'):
            self.__adjacency_matrix_ctr(min_count)
            return self.matrix_a
        elif (entry == 'inst'):
            self.__adjacency_matrix_inst(min_count)
            return self.matrix_a
        elif (entry == 'kwa'):
            self.__adjacency_matrix_kwa(min_count)
            return self.matrix_a
        elif (entry == 'kwp'):
            self.__adjacency_matrix_kwp(min_count)
            return self.matrix_a
        elif (entry == 'ref'):
            self.__adjacency_matrix_ref(min_count, local_nodes)
            return self.matrix_r

    # Function: Network Collab
    def network_collab(self, entry = 'aut', tgt = [], topn = 15, rows = 5, cols = 3, wspace = 0.2, hspace = 0.2, tspace = 0.01, node_size = 300, font_size = 8, pad = 0.2, nd_a = '#FF0000', nd_b = '#008000', nd_c = '#808080', verbose = False):
        if (entry == 'aut'):
            self.__adjacency_matrix_aut(0)
            collab_data = self.matrix_a.copy(deep = True)
            targets     = [item for item in self.u_aut]
            sizes       = self.doc_aut
        if (entry == 'cout'):
            self.__adjacency_matrix_ctr(0) 
            collab_data = self.matrix_a.copy(deep = True)
            targets     = [item for item in self.u_ctr]
            sizes       = self.__get_counts(self.u_ctr, self.ctr)
        if (entry == 'inst'):
            self.__adjacency_matrix_inst(0) 
            collab_data = self.matrix_a.copy(deep = True)
            targets     = [item for item in self.u_uni]
            sizes       = self.__get_counts(self.u_uni, self.uni)
        if (entry == 'kwa'):
            self.__adjacency_matrix_kwa(0) 
            collab_data = self.matrix_a.copy(deep = True)
            targets     = [item for item in self.u_auk]
            sizes       = self.__get_counts(self.u_auk, self.auk)
        if (entry == 'kwp'):
            self.__adjacency_matrix_kwp(0)
            collab_data = self.matrix_a.copy(deep = True)
            targets     = [item for item in self.u_kid]
            sizes       = self.__get_counts(self.u_kid, self.kid)
        collab_data     = collab_data.reset_index(drop = True)
        if (len(tgt) == 0):
            idx     = sorted(range(len(sizes)), key = sizes.__getitem__)
            idx.reverse()
            targets = [targets[i] for i in idx]
            targets = targets[:topn]
        else:
            targets = [item for item in tgt]
        highlight_list  = set(targets) if len(targets) > 1 else set()
        self.ask_gpt_ct = []
        if (len(targets) == 1):
            rows, cols = 1, 1
        elif (len(targets) > rows * cols):
            rows = int(np.ceil(len(targets) / cols))
        fig, axes = plt.subplots(rows, cols, figsize = (cols * 5, rows * 5), facecolor = 'white')
        axes      = axes.flatten() if len(targets) > 1 else [axes]
        for idx, target in enumerate(targets):
            ax          = axes[idx]
            G           = nx.Graph()
            G.add_node(target, color = nd_a)
            connections = collab_data[collab_data[target] > 0].index.tolist()
            ct          = [collab_data.columns[conn_idx] for conn_idx in connections]
            self.ask_gpt_ct.append([ [target], ct ])
            if (verbose == True):
                print(f'Main Node: {target}')
                print(f'Links: {ct}\n')
            for conn_idx in connections:
                conn_target = collab_data.columns[conn_idx]
                color       = nd_b if conn_target in highlight_list else nd_c
                G.add_node(conn_target, color = color)
                G.add_edge(target, conn_target, color = color)
            node_colors = [G.nodes[n]['color'] for n in G.nodes()]
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            pos         = nx.spring_layout(G, seed=42)
            label_pos   = {k: (v[0], v[1] + tspace) for k, v in pos.items()}
            labels      = {node: '' if G.nodes[node]['color'] == nd_a else node for node in G.nodes()}
            nx.draw_networkx_nodes(G, pos, node_color = node_colors, node_size = node_size, ax = ax)
            nx.draw_networkx_edges(G, pos, edge_color = edge_colors, ax = ax)
            nx.draw_networkx_labels(G, label_pos, labels, font_size = font_size, ax = ax)
            ax.set_title(target.title(), color = nd_a, fontsize = font_size + 1)
            ax.axis('off')
            rect = plt.Rectangle((0, 0), 1, 1, fill = False, edgecolor = 'black', linewidth = 0.5, transform = ax.transAxes, clip_on = False)
            ax.add_patch(rect)
        for j in range(len(targets), rows * cols):
            fig.delaxes(axes[j])
        plt.subplots_adjust(wspace = wspace, hspace = hspace)
        plt.tight_layout(pad = pad)
        plt.show()
        return

    # Function: Network Similarities 
    def network_sim(self, view = 'browser', sim_type = 'coup', node_size = -1, node_labels = False, cut_coup = 0.3, cut_cocit = 5):
        sim = ''
        if   (sim_type == 'coup'):
            cut = cut_coup
            sim = 'Bibliographic Coupling'
        elif (sim_type == 'cocit'):
            cut = cut_cocit
            sim = 'Co-Citation'
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_labels == True):
            mode = 'markers+text'
            size = 17
        else:
            mode = 'markers'
            size = 10
        if (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        self.__adjacency_matrix_ref(1, False)
        adjacency_matrix = self.matrix_r.values
        if (sim_type == 'coup'):
            adjacency_matrix = cosine_similarity(adjacency_matrix)
        elif (sim_type == 'cocit'):
            x = pd.DataFrame(np.zeros((adjacency_matrix.shape[0], adjacency_matrix.shape[0])))
            for i in range(0, adjacency_matrix.shape[0]):
                for j in range(0, adjacency_matrix.shape[0]):
                    if (i != j):
                        ct = adjacency_matrix[i,:] + adjacency_matrix[j,:]
                        ct = ct.tolist()
                        vl = ct.count(2.0)
                        if (vl > 0):
                            x.iloc[i, j] = vl
            adjacency_matrix = x
        adjacency_matrix = np.triu(adjacency_matrix, k = 1)
        S                = nx.Graph()
        rows, cols       = np.where(adjacency_matrix >= cut)
        edges            = list(zip(rows.tolist(), cols.tolist()))
        u_rows           = list(set(rows.tolist()))
        u_rows           = [str(item) for item in u_rows]
        u_rows           = sorted(u_rows, key = self.natsort)
        u_cols           = list(set(cols.tolist()))
        u_cols           = [str(item) for item in u_cols]
        u_cols           = sorted(u_cols, key = self.natsort)
        for name in u_rows:
            color = 'blue'
            year  = int(self.dy[ int(name) ])
            n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
            S.add_node(name, color = color, year = year, n_id = n_id )
        for name in u_cols:
            if (name not in u_rows):
                color = 'blue'
                year  = int(self.dy[ int(name) ])
                n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
                S.add_node(name, color = color, year = year, n_id = n_id )
        self.sim_table   = pd.DataFrame(np.zeros((len(edges), 2)), columns = ['Pair Node', 'Sim('+sim_type+')'])
        self.ask_gpt_sim = pd.DataFrame(np.zeros((len(edges), 3)), columns = ['Node 1', 'Node 2', 'Simimilarity ('+sim+') Between Nodes'])
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = str(srt)
            end_     = str(end)
            if ( end_ != '-1' ):
                wght = round(adjacency_matrix[srt, end], 3)
                S.add_edge(srt_, end_, weight = wght)
                self.sim_table.iloc[i, 0]   = '('+srt_+','+end_+')'
                self.sim_table.iloc[i, 1]   = wght
                self.ask_gpt_sim.iloc[i, 0] = 'Paper ID: '+srt_
                self.ask_gpt_sim.iloc[i, 1] = 'Paper ID: '+end_
                self.ask_gpt_sim.iloc[i, 2] = wght
        generator      = nx.algorithms.community.girvan_newman(S)
        community      = next(generator)
        community_list = sorted(map(sorted, community))
        for com in community_list:
            community_list.index(com)
            for node in com:
                S.nodes[node]['color'] = self.color_names[community_list.index(com)]
                S.nodes[node]['n_cls'] = community_list.index(com)
        color       = [S.nodes[n]['color'] for n in S.nodes()]
        pos_s       = nx.spring_layout(S, seed = 42, scale = 1)
        node_list_s = list(S.nodes)
        edge_list_s = list(S.edges)
        nids_list_s = [S.nodes[n]['n_id'] for n in S.nodes()]
        nids_list_s = ['<br>'.join(textwrap.wrap(txt, width = 50)) for txt in nids_list_s]
        nids_list_s = ['id: '+node_list_s[i]+'<br>'+nids_list_s[i] for i in range(0, len(nids_list_s))]
        Xw          = [pos_s[k][0] for k in node_list_s]
        Yw          = [pos_s[k][1] for k in node_list_s]
        Xi          = []
        Yi          = []
        for edge in edge_list_s:
            Xi.append(pos_s[edge[0]][0]*1.00)
            Xi.append(pos_s[edge[1]][0]*1.00)
            Xi.append(None)
            Yi.append(pos_s[edge[0]][1]*1.00)
            Yi.append(pos_s[edge[1]][1]*1.00)
            Yi.append(None)
        a_trace = go.Scatter(x         = Xi,
                             y         = Yi,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = Xw,
                             y         = Yw,
                             opacity   = 0.45,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = color, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = node_list_s,
                             hoverinfo = 'text',
                             hovertext = nids_list_s,
                             name      = ''
                             )
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = False),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            )
        fig_s = go.Figure(data = [n_trace, a_trace], layout = layout)
        fig_s.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        fig_s.update_traces(textfont_size = 10, textfont_color = 'blue', textposition = 'top center') 
        fig_s.show()  
        return

    # Function: Map from Country Adjacency Matrix
    def network_adj_map(self, view = 'browser', connections = True, country_lst = []):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        lat_             = [self.country_lat_long[i][0] for i in range(0, len(self.country_lat_long)) if self.country_names[i] in self.u_ctr]
        lon_             = [self.country_lat_long[i][1] for i in range(0, len(self.country_lat_long)) if self.country_names[i] in self.u_ctr]
        iso_3            = [self.country_alpha_3[i] for i in range(0, len(self.country_lat_long)) if self.country_names[i] in self.u_ctr]
        text             = [item for item in self.country_names if item in self.u_ctr]
        self.__adjacency_matrix_ctr(1)
        adjacency_matrix = self.matrix_a.values
        try:
            row_pos                      = self.matrix_a.index.get_loc('UNKNOWN')
            col_pos                      = self.matrix_a.columns.get_loc('UNKNOWN')
            adjacency_matrix[row_pos, :] = 0
            adjacency_matrix[:, col_pos] = 0
        except:
            pass
        vals             = [ int(self.dict_ctr_id[text[i]].replace('c_','')) for i in range(0, len(text))]
        vals             = [ int(np.sum(adjacency_matrix[i,:])) for i in vals ]
        lat_             = [ lat_[i] for i in range(0, len(vals)) if vals[i] > 0]
        lon_             = [ lon_[i] for i in range(0, len(vals)) if vals[i] > 0]
        iso_3            = [iso_3[i] for i in range(0, len(vals)) if vals[i] > 0]
        text             = [ text[i] for i in range(0, len(vals)) if vals[i] > 0]
        vals             = [ vals[i] for i in range(0, len(vals)) if vals[i] > 0]
        rows, cols       = np.where(adjacency_matrix >= 1)
        edges            = list(zip(rows.tolist(), cols.tolist()))
        try:
            unk   = int(self.dict_ctr_id['UNKNOWN'].replace('c_',''))
            edges = list(filter(lambda edge: unk not in edge, edges))
        except:
            pass
        self.ask_gpt_map = pd.DataFrame(edges, columns = ['Country 1', 'Country 2']) 
        nids_list  = ['id:                        ' +self.dict_ctr_id[text[i]]+'<br>'+
                      'country:               '     +text[i].upper()+'<br>' +
                      'collaborators:      '        +str(vals[i])  
                      for i in range(0, len(lat_))]
        Xa         = []
        Ya         = []
        Xb         = []
        Yb         = []
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt      = 'c_'+str(srt)
            end      = 'c_'+str(end)
            srt      = self.dict_id_ctr[srt]
            end      = self.dict_id_ctr[end]
            self.ask_gpt_map.iloc[i, 0] = srt
            self.ask_gpt_map.iloc[i, 1] = end
            if (len(country_lst) > 0):
                country_lst = [item.lower() for item in country_lst]
                for j in range(0, len(country_lst)):
                    if (srt.lower() in country_lst or end.lower() in country_lst):
                        srt_ = self.country_names.index(srt)
                        end_ = self.country_names.index(end)
                        Xb.append(self.country_lat_long[srt_][0]) 
                        Xb.append(self.country_lat_long[end_][0]) 
                        Xb.append(None)
                        Yb.append(self.country_lat_long[srt_][1])
                        Yb.append(self.country_lat_long[end_][1])
                        Yb.append(None)
            srt = self.country_names.index(srt)
            end = self.country_names.index(end)
            Xa.append(self.country_lat_long[srt][0]) 
            Xa.append(self.country_lat_long[end][0]) 
            Xa.append(None)
            Ya.append(self.country_lat_long[srt][1])
            Ya.append(self.country_lat_long[end][1])
            Ya.append(None) 
        data   = dict(type                  = 'choropleth',
                      locations             = iso_3,
                      locationmode          = 'ISO-3',
                      colorscale            = 'sunsetdark', 
                      z                     = vals,
                      hoverinfo             = 'none'
                      )
        edges  = go.Scattergeo(lat          = Xa,
                               lon          = Ya,
                               mode         = 'lines',
                               line         = dict(color = 'rgba(15, 84, 26, 0.25)', width = 1, dash = 'solid'),
                               hoverinfo    = 'none',
                               name         = ''
                               )
        edge_h = go.Scattergeo(lat          = Xb,
                               lon          = Yb,
                               mode         = 'lines',
                               line         = dict(color = 'rgba(255, 3, 45, 0.85)', width = 1, dash = 'solid'),
                               hoverinfo    = 'none',
                               name         = ''
                               )
        nodes  = go.Scattergeo(lon          = lon_,
                               lat          = lat_,
                               text         = text,
                               textfont     = dict(color = 'black', family =  'Times New Roman', size = 10),
                               textposition = 'top center',
                               mode         = 'markers+text',
                               marker       = dict(size = 7, color = 'white', line_color = 'black', line_width = 1),
                               hoverinfo    = 'text',
                               hovertext    = nids_list,
                               name         = '',
                               )
        layout = dict(geo = {'scope': 'world'}, showlegend = False, hovermode  = 'closest',  hoverlabel = dict(font_size = 12), margin = dict(b = 10, l = 5, r = 5, t = 10))
        if (connections == True):
            geo_data = [data, edges]
        else:
            geo_data = [data]
        if (len(country_lst) > 0):
           geo_data.append(edge_h)
        geo_data.append(nodes)
        fig_cm = go.Figure(data = geo_data, layout = layout)
        fig_cm.update_geos(resolution     = 50,
                        showcoastlines = True,  coastlinecolor = 'black',
                        showland       = True,  landcolor      = '#f0f0f0',
                        showocean      = True,  oceancolor     = '#7fcdff',  # '#def3f6', '#7fcdff',
                        showlakes      = False, lakecolor      = 'blue',
                        showrivers     = False, rivercolor     = 'blue',
            			lataxis        = dict(  range          = [-60, 90]), # clip Antarctica
                        )   
        fig_cm.show()
        return

    # Function: Direct Network from Adjacency Matrix
    def network_adj_dir(self, view = 'browser', min_count = 1, node_size = -1, font_size = 10, node_labels = False, local_nodes = False):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_labels == True and node_size == -1):
            mode = 'markers+text'
            size = 50
        elif (node_labels == False and node_size == -1):
            mode = 'markers'
            size = 10
        elif (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        self.__adjacency_matrix_ref(min_count, local_nodes)
        adjacency_matrix = self.matrix_r.values
        G                = nx.DiGraph()
        rows, cols       = np.where(adjacency_matrix >= 1)
        edges            = list(zip(rows.tolist(), cols.tolist()))
        u_rows           = list(set(rows.tolist()))
        u_cols           = list(set(cols.tolist()))
        labels           = [self.labels_r[item] for item in u_cols]
        labels           = sorted(labels, key = self.natsort)
        for name in labels: 
            if (name.find('r_') != -1):
                color = 'red'
                year  = self.dy_ref[ int(name.replace('r_','')) ]
                if (len(self.u_ref) > 0):
                    n_id  = self.u_ref [ int(name.replace('r_','')) ]
                else:
                    n_id  = ''
                G.add_node(name, color = color,  year = year, n_id = n_id)
            else:
                if (int(name.replace('r_','')) not in u_rows):
                    u_rows.append(int(name.replace('r_','')))
        u_rows = [str(item) for item in u_rows]
        u_rows = sorted(u_rows, key = self.natsort)
        for name in u_rows:
            color = 'blue'
            year  = int(self.dy[ int(name) ])
            n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
            G.add_node(name, color = color, year = year, n_id = n_id )
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = str(srt)
            end_     = self.labels_r[end]
            if ( end_ != '-1' ):
                G.add_edge(srt_, end_)
        self.ask_gpt_nad = pd.DataFrame(G.edges, columns = ['Paper', 'Cited Reference'])
        color            = [G.nodes[n]['color'] if len(G.nodes[n]) > 0 else 'black' for n in G.nodes()]
        self.pos         = nx.circular_layout(G)
        self.node_list   = list(G.nodes)
        self.edge_list   = list(G.edges)
        self.nids_list   = [G.nodes[n]['n_id'] for n in G.nodes()]
        self.nids_list   = ['<br>'.join(textwrap.wrap(txt, width = 50)) for txt in self.nids_list]
        self.nids_list   = ['id: '+self.node_list[i]+'<br>'+self.nids_list[i] for i in range(0, len(self.nids_list))]
        self.Xn          = [self.pos[k][0] for k in self.node_list]
        self.Yn          = [self.pos[k][1] for k in self.node_list]
        Xa               = []
        Ya               = []
        for edge in self.edge_list:
            Xa.append(self.pos[edge[0]][0]*0.97)
            Xa.append(self.pos[edge[1]][0]*0.97)
            Xa.append(None)
            Ya.append(self.pos[edge[0]][1]*0.97)
            Ya.append(self.pos[edge[1]][1]*0.97)
            Ya.append(None)
        a_trace = go.Scatter(x         = Xa,
                             y         = Ya,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'dash'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = self.Xn,
                             y         = self.Yn,
                             opacity   = 0.45,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = color, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = self.node_list,
                             hoverinfo = 'text',
                             hovertext = self.nids_list,
                             name      = ''
                             )
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = False),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            )
        self.fig = go.Figure(data = [n_trace, a_trace], layout = layout)
        self.fig.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        self.fig.update_traces(textfont_size = font_size, textfont_color = 'yellow') 
        self.fig.show()
        return

    # Function: Network from Adjacency Matrix 
    def network_adj(self, view = 'browser', adj_type = 'aut', min_count = 2, node_size = -1, font_size = 10, node_labels = False, label_type = 'id', centrality = None): 
        adj_ = ''
        cen_ = 'Girvan-Newman Community Algorithm'
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if (node_labels == True):
            mode = 'markers+text'
            size = 17
        else:
            mode = 'markers'
            size = 10
        if (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        if   (adj_type == 'aut'):
            self.__adjacency_matrix_aut(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_aut
            adj_             = 'Author'
        elif (adj_type == 'cout'):
            self.__adjacency_matrix_ctr(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_ctr
            adj_             = 'Country'
        elif (adj_type == 'inst'):
            self.__adjacency_matrix_inst(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_uni
            adj_             = 'Institution'
        elif (adj_type == 'kwa'):
            self.__adjacency_matrix_kwa(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_kwa
            adj_             = 'Author Keywords'
        elif (adj_type == 'kwp'):
            self.__adjacency_matrix_kwp(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_kwp
            adj_             = 'Keywords Plus'
        rows, cols = np.where(adjacency_matrix >= 1)
        edges      = list(zip(rows.tolist(), cols.tolist()))
        u_cols     = list(set(cols.tolist()))
        self.H     = nx.Graph()
        if (adj_type == 'aut'):
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[u_cols[i]]
                n_doc = self.doc_aut[int(name.replace('a_',''))]
                n_lhi = self.aut_h[int(name.replace('a_',''))]
                n_id  = self.u_aut[int(name.replace('a_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_doc = n_doc, n_lhi = n_lhi, n_id = n_id )
        elif (adj_type == 'cout'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[u_cols[i]]
                n_id  = self.u_ctr[int(name.replace('c_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        elif (adj_type == 'inst'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[u_cols[i]]
                n_id  = self.u_uni[int(name.replace('i_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        elif (adj_type == 'kwa'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[u_cols[i]]
                n_id  = self.u_auk[int(name.replace('k_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        elif (adj_type == 'kwp'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[u_cols[i]]
                n_id  = self.u_kid[int(name.replace('p_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = self.labels_a[srt]
            end_     = self.labels_a[end]
            if ( end_ != '-1'):
                self.H.add_edge(srt_, end_)
        dict_cen = []
        if (centrality == 'degree'): 
            value            = nx.algorithms.centrality.degree_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Degree'])
            self.table_centr = self.table_centr.sort_values('Degree', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Degree Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        elif (centrality == 'load'):
            value            = nx.algorithms.centrality.load_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Load'])
            self.table_centr = self.table_centr.sort_values('Load', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Load Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        elif (centrality == 'betw'):
            value            = nx.algorithms.centrality.betweenness_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Betweenness'])
            self.table_centr = self.table_centr.sort_values('Betweenness', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Betweenness Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        elif (centrality == 'close'):
            value            = nx.algorithms.centrality.closeness_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Closeness'])
            self.table_centr = self.table_centr.sort_values('Closeness', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Closeness Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        elif (centrality == 'eigen'):
            value            = nx.algorithms.centrality.eigenvector_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Eigenvector'])
            self.table_centr = self.table_centr.sort_values('Eigenvector', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Eigenvector Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        elif (centrality == 'katz'):
            value            = nx.algorithms.centrality.katz_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Katz'])
            self.table_centr = self.table_centr.sort_values('Katz', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Katz Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        elif (centrality == 'harmonic'):
            value            = nx.algorithms.centrality.harmonic_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Harmonic'])
            self.table_centr = self.table_centr.sort_values('Harmonic', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
            cen_             = 'Harmonic Centrality'
            dict_cen         = dict(zip(self.table_centr.iloc[:,-2], self.table_centr.iloc[:,-1]))
        else:
            generator        = nx.algorithms.community.girvan_newman(self.H)
            community        = next(generator)
            community_list   = sorted(map(sorted, community))
            for com in community_list:
                community_list.index(com)
                for node in com:
                    self.H.nodes[node]['color'] = self.color_names[community_list.index(com)]
                    self.H.nodes[node]['n_cls'] = community_list.index(com)
            color = [self.H.nodes[n]['color'] for n in self.H.nodes()]
        self.pos_a       = nx.spring_layout(self.H, seed = 42, scale = 1000)
        self.node_list_a = list(self.H.nodes)
        self.edge_list_a = list(self.H.edges)
        if (cen_ == 'Girvan-Newman Community Algorithm'):
            self.ask_gpt_adj = pd.DataFrame((np.zeros((len(self.H.edges), 4))), columns = ['Node 1'+' ('+adj_+')', 'Node 2'+' ('+adj_+')', 'Node 1 Cluster', 'Node 2 Cluster'])
        else:
            self.ask_gpt_adj = pd.DataFrame((np.zeros((len(self.H.edges), 4))), columns = ['Node 1'+' ('+adj_+')', 'Node 2'+' ('+adj_+')', 'Node 1' + ' ('+cen_+')', 'Node 2' + ' ('+cen_+')'])
        if (cen_ == 'Girvan-Newman Community Algorithm'):
            for i in range(0, self.ask_gpt_adj.shape[0]):
                srt, end                    = list(self.H.edges)[i]
                self.ask_gpt_adj.iloc[i, 0] = 'ID: ' + srt
                self.ask_gpt_adj.iloc[i, 1] = 'ID: ' + end
                self.ask_gpt_adj.iloc[i, 2] = self.H.nodes[srt]['n_cls']
                self.ask_gpt_adj.iloc[i, 3] = self.H.nodes[end]['n_cls']
        else:
            for i in range(0, self.ask_gpt_adj.shape[0]):
                srt, end                    = list(self.H.edges)[i]
                self.ask_gpt_adj.iloc[i, 0] = 'ID: ' + srt
                self.ask_gpt_adj.iloc[i, 1] = 'ID: ' + end
                self.ask_gpt_adj.iloc[i, 2] = dict_cen[srt]
                self.ask_gpt_adj.iloc[i, 3] = dict_cen[end]    
        if (adj_type == 'aut'):
            docs_list_a      = [self.H.nodes[n]['n_doc'] for n in self.H.nodes()]
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            lhid_list_a      = [self.H.nodes[n]['n_lhi'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                       ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '   +str(clst_list_a[i])+'<br>'+
                                'author:                '    +self.nids_list_a[i].upper()+'<br>'+
                                'documents:         '        +str(docs_list_a[i])+'<br>'+
                                'collaborators:      '       +str(auts_list_a[i])+'<br>'+
                                'local h-index:       '      +str(lhid_list_a[i]) 
                                for i in range(0, len(self.nids_list_a))]     
        elif (adj_type == 'cout'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'country:               '     +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'inst'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'institution:            '    +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'kwa'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'author keyword:    '         +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'kwp'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'keyword plus:     '          +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        self.Xv = [self.pos_a[k][0] for k in self.node_list_a]
        self.Yv = [self.pos_a[k][1] for k in self.node_list_a]
        Xe      = []
        Ye      = []
        if (label_type != 'id'):
            if (adj_type == 'aut'):
                self.node_list_a = [ self.dict_id_aut[item] for item in self.node_list_a]
            elif (adj_type == 'cout'):
                self.node_list_a = [ self.dict_id_ctr[item] for item in self.node_list_a]
            elif (adj_type == 'inst'): 
                self.node_list_a = [ self.dict_id_uni[item] for item in self.node_list_a]
            elif (adj_type == 'kwa'):
                self.node_list_a = [ self.dict_id_kwa[item] for item in self.node_list_a]
            elif (adj_type == 'kwp'): 
                self.node_list_a = [ self.dict_id_kwp[item] for item in self.node_list_a]
        for edge in self.edge_list_a:
            Xe.append(self.pos_a[edge[0]][0]*1.00)
            Xe.append(self.pos_a[edge[1]][0]*1.00)
            Xe.append(None)
            Ye.append(self.pos_a[edge[0]][1]*1.00)
            Ye.append(self.pos_a[edge[1]][1]*1.00)
            Ye.append(None)
        a_trace = go.Scatter(x         = Xe,
                             y         = Ye,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = self.Xv,
                             y         = self.Yv,
                             opacity   = 0.57,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = color, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = self.node_list_a,
                             hoverinfo = 'text',
                             hovertext = self.nids_list_a,
                             name      = ''
                             )
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = False),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            )
        self.fig_a = go.Figure(data = [n_trace, a_trace], layout = layout)
        self.fig_a.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        self.fig_a.update_traces(textfont_size = font_size, textfont_color = 'blue', textposition = 'top center') 
        self.fig_a.show()
        if (label_type != 'id'):
            if (adj_type == 'aut'):
                self.node_list_a = [ self.dict_aut_id[item] for item in self.node_list_a]
            elif (adj_type == 'cout'):
                self.node_list_a = [ self.dict_ctr_id[item] for item in self.node_list_a]
            elif (adj_type == 'inst'): 
                self.node_list_a = [ self.dict_uni_id[item] for item in self.node_list_a]
            elif (adj_type == 'kwa'):
                self.node_list_a = [ self.dict_kwa_id[item] for item in self.node_list_a]
            elif (adj_type == 'kwp'): 
                self.node_list_a = [ self.dict_kwp_id[item] for item in self.node_list_a]
        return

    # Function: Find Connected Nodes from Direct Network
    def find_nodes_dir(self, view = 'browser', article_ids = [], ref_ids = [], node_size = -1, font_size = 10):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_size > 0):
            size = node_size
        else:
            size = 50
        fig_ = go.Figure(self.fig)
        if (len(article_ids) > 0 or len(ref_ids) > 0):
            if (len(article_ids) > 0):
                edge_list_ai = []
                idx_ids      = []
                color_ids    = []
                text_ids     = []
                hover_ids    = []
                article_ids  = [str(int(item)) for item in article_ids]
                idx_ids.extend([self.node_list.index(node) for node in article_ids])
                color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in article_ids])
                text_ids.extend([node for node in article_ids])
                hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in article_ids])
                for ids in article_ids:
                    edge_list_ai.extend([item for item in self.edge_list if ids == item[0]])
                    node_pair = [self.edge_list[i][1] for i in range(0, len(self.edge_list)) if ids in self.edge_list[i]]
                    idx_ids.extend([self.node_list.index(node) for node in node_pair])
                    color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in node_pair])
                    text_ids.extend([node for node in node_pair])
                    hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in node_pair])
                xa = []
                ya = []
                if (len(edge_list_ai) > 0):
                    for edge in edge_list_ai:
                        xa.append(self.pos[edge[0]][0]*0.97)
                        xa.append(self.pos[edge[1]][0]*0.97)
                        ya.append(self.pos[edge[0]][1]*0.97)
                        ya.append(self.pos[edge[1]][1]*0.97)
                    for i in range(0, len(xa), 2):
                        fig_.add_annotation(
                                           x          = xa[i + 1],  # to x
                                           y          = ya[i + 1],  # to y
                                           ax         = xa[i + 0],  # from x
                                           ay         = ya[i + 0],  # from y
                                           xref       = 'x',
                                           yref       = 'y',
                                           axref      = 'x',
                                           ayref      = 'y',
                                           text       = '',
                                           showarrow  = True,
                                           arrowhead  = 3,
                                           arrowsize  = 1.2,
                                           arrowwidth = 1,
                                           arrowcolor = 'black',
                                           opacity    = 0.9
                                           )
                xn = [self.Xn[i] for i in idx_ids]
                yn = [self.Yn[i] for i in idx_ids]
                fig_.add_trace(go.Scatter(x              = xn,
                                          y              = yn,
                                          mode           = 'markers+text',
                                          marker         = dict(symbol = 'circle-dot', size = size, color = color_ids),
                                          text           = text_ids,
                                          hoverinfo      = 'text',
                                          hovertext      = hover_ids,
                                          textfont_size  = 10, 
                                          textfont_color = 'yellow',
                                          name           = ''
                                          ))
            if (len(ref_ids) > 0):
                edge_list_ri = []
                idx_ids      = []
                color_ids    = []
                text_ids     = []
                hover_ids    = []
                ref_ids      = [str(item) for item in ref_ids]
                idx_ids.extend([self.node_list.index(node) for node in ref_ids])
                color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in ref_ids])
                text_ids.extend([node for node in ref_ids])
                hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in ref_ids])
                for ids in ref_ids:
                    edge_list_ri.extend([item for item in self.edge_list if ids == item[1]]) 
                    node_pair = [self.edge_list[i][0] for i in range(0, len(self.edge_list)) if ids in self.edge_list[i]]
                    idx_ids.extend([self.node_list.index(node) for node in node_pair])
                    color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in node_pair])
                    text_ids.extend([node for node in node_pair])
                    hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in node_pair])
                xa = []
                ya = []
                if (len(edge_list_ri) > 0):
                    for edge in edge_list_ri:
                        xa.append(self.pos[edge[0]][0]*0.97)
                        xa.append(self.pos[edge[1]][0]*0.97)
                        ya.append(self.pos[edge[0]][1]*0.97)
                        ya.append(self.pos[edge[1]][1]*0.97)
                    for i in range(0, len(xa), 2):
                        fig_.add_annotation(
                                           x          = xa[i + 1],  # to x
                                           y          = ya[i + 1],  # to y
                                           ax         = xa[i + 0],  # from x
                                           ay         = ya[i + 0],  # from y
                                           xref       = 'x',
                                           yref       = 'y',
                                           axref      = 'x',
                                           ayref      = 'y',
                                           text       = '',
                                           showarrow  = True,
                                           arrowhead  = 3,
                                           arrowsize  = 1.2,
                                           arrowwidth = 1,
                                           arrowcolor = 'black',
                                           opacity    = 0.9
                                           )
                xn = [self.Xn[i] for i in idx_ids]
                yn = [self.Yn[i] for i in idx_ids]
                fig_.add_trace(go.Scatter(x              = xn,
                                          y              = yn,
                                          mode           = 'markers+text',
                                          marker         = dict(symbol = 'circle-dot', size = size, color = color_ids),
                                          text           = text_ids,
                                          hoverinfo      = 'text',
                                          hovertext      = hover_ids,
                                          textfont_size  = font_size, 
                                          textfont_color = 'yellow',
                                          name           = ''
                                          ))
        fig_.show()
        return 

    # Function: Find Connected Nodes
    def find_nodes(self, node_ids = [], node_name = [], node_size = -1, font_size = 10, node_only = False):
        flag = False
        if (len(node_ids) == 0 and len(node_name) > 0):
            if   (node_name[0] in self.dict_aut_id.keys()):
                node_ids = [self.dict_aut_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_ctr_id.keys()):
                node_ids = [self.dict_ctr_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_uni_id.keys()):
                node_ids = [self.dict_uni_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_kwa_id.keys()):
                node_ids = [self.dict_kwa_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_kwp_id.keys()):
                node_ids = [self.dict_kwp_id[item] for item in node_name]
                flag     = True
        if (node_size > 0):
            size = node_size
        else:
            size = 17
        fig_ = go.Figure(self.fig_a)
        fig_.update_traces(mode = 'markers', line = dict(width = 0), marker = dict(color = 'rgba(0, 0, 0, 0.1)', size = 7)) 
        if (len(node_ids) > 0):
            edge_list_ai = []
            idx_ids      = []
            color_ids    = []
            text_ids     = []
            hover_ids    = []
            idx_ids.extend([self.node_list_a.index(node) for node in node_ids])
            color_ids.extend(['black' for n in node_ids]) # self.H.nodes[n]['color']
            text_ids.extend([node for node in node_ids])
            hover_ids.extend([self.nids_list_a[self.node_list_a.index(node)] for node in node_ids])
            if (node_only != True):
                for ids in node_ids:
                    edge_list_ai.extend([item for item in self.edge_list_a if ids in item and item not in edge_list_ai])
                    node_pair = [self.edge_list_a[i][1] for i in range(0, len(self.edge_list_a)) if ids in self.edge_list_a[i] and self.edge_list_a[i][1] != ids]
                    node_pair.extend([self.edge_list_a[i][0] for i in range(0, len(self.edge_list_a)) if ids in self.edge_list_a[i] and self.edge_list_a[i][0] != ids])
                    idx_ids.extend([self.node_list_a.index(node) for node in node_pair])
                    color_ids.extend(['#e0cc92' for n in node_pair]) # self.H.nodes[n]['color']
                    text_ids.extend([node for node in node_pair])
                    hover_ids.extend([self.nids_list_a[self.node_list_a.index(node)] for node in node_pair])
                xa = []
                ya = []
                for edge in edge_list_ai:
                    xa.append(self.pos_a[edge[0]][0]*1.00)
                    xa.append(self.pos_a[edge[1]][0]*1.00)
                    xa.append(None)
                    ya.append(self.pos_a[edge[0]][1]*1.00)
                    ya.append(self.pos_a[edge[1]][1]*1.00)
                    ya.append(None)
                for i in range(0, len(xa), 2):
                    fig_.add_trace(go.Scatter(x         = xa,
                                              y         = ya,
                                              mode      = 'lines',
                                              line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'solid'),
                                              hoverinfo = 'none',
                                              name      = ''
                                              ))
            xn = [self.Xv[i] for i in idx_ids]
            yn = [self.Yv[i] for i in idx_ids]
            if (flag == True):
                if   (node_name[0] in self.dict_aut_id.keys()):
                    text_ids = [self.dict_id_aut[item] for item in text_ids]
                elif (node_name[0] in self.dict_ctr_id.keys()):
                    text_ids = [self.dict_id_ctr[item] for item in text_ids]
                elif (node_name[0] in self.dict_uni_id.keys()):
                    text_ids = [self.dict_id_uni[item] for item in text_ids]
                elif (node_name[0] in self.dict_kwa_id.keys()):
                    text_ids = [self.dict_id_kwa[item] for item in text_ids]
                elif (node_name[0] in self.dict_kwp_id.keys()):
                    text_ids = [self.dict_id_kwp[item] for item in text_ids]
            fig_.add_trace(go.Scatter(x              = xn,
                                      y              = yn,
                                      mode           = 'markers+text',
                                      marker         = dict(symbol = 'circle-dot', size = size, color = color_ids),
                                      text           = text_ids,
                                      hoverinfo      = 'text',
                                      hovertext      = hover_ids,
                                      textfont_size  = font_size, 
                                      name           = ''
                                      ))
        fig_.update_traces(textposition = 'top center') 
        fig_.show()
        return
    
    # Function: Citation History Network
    def network_hist(self, view = 'browser', min_links = 0, chain = [], path = True, node_size = 20, font_size = 10, node_labels = True, dist = 1.2, dist_pad = 0):
        
        #----------------------------------------------------------------------
                     
        def filter_matrix_by_citations(matrix_r, min_links):
            filtered_matrix = matrix_r.copy()
            row_sums        = filtered_matrix.sum(axis = 1)
            row_sums_sorted = row_sums.sort_values(ascending = False) 
            hold_list       = set() 
            node_min        = []
            for node, link_count in row_sums_sorted.items():
                if (link_count >= min_links):
                    hold_list.add(node)
                    node_min.append(node)
                    linked_nodes = filtered_matrix.loc[node][filtered_matrix.loc[node] > 0].index
                    linked_nodes = [int(item) for item in linked_nodes ]
                    hold_list.update(linked_nodes)                     
            node_min        = [str(item) for item in node_min]
            all_nodes       = set(filtered_matrix.index)
            dropped_indices = list(all_nodes - hold_list)
            hold_list       = list(hold_list)
            filtered_matrix = filtered_matrix.loc[hold_list, :]
            return filtered_matrix, dropped_indices, node_min

         #----------------------------------------------------------------------
        
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if (node_labels == True):
            mode = 'markers+text'
        else:
            mode = 'markers'
        if (len(chain) > 0):
            chain = [str(c) for c in chain]
        articles          = self.data[['author', 'title',  'journal', 'doi']]
        articles['id']    = self.table_id_doc.iloc[:,0]
        articles['year']  = self.data['year'].astype(int)
        self.__adjacency_matrix_ref(0, True)
        matrix, idx, n_m  = filter_matrix_by_citations(self.matrix_r, min_links)
        rows, cols        = np.where(matrix >= 1)
        row_map           = {idx: int(row_name) for idx, row_name in enumerate(matrix.index)}
        col_map           = {idx: int(col_name) for idx, col_name in enumerate(matrix.columns)}
        citations         = [(row_map[r], col_map[c]) for (r, c) in zip(rows, cols)]
        rev               = [(tgt, src) for (src, tgt) in citations]
        citations         = citations + rev    
        citations         = [tup for tup in citations if tup[0] != tup[1]]
        articles          = articles.drop(index = idx, errors = 'ignore')
        articles          = articles.sort_values(by = 'year').reset_index(drop = True)
        articles['x_pos'] = articles['year']
        years             = sorted(articles['year'].unique())
        y_offsets         = {}
        for yr in years:
            indices = articles.index[articles['year'] == yr].tolist()
            indices = sorted(indices, key = lambda x: self.natsort(str(articles.loc[x, 'id'])))
            count   = len(indices)
            offsets = [i*dist for i in range(0, count)]
            for idx, art_idx in enumerate(indices):
                y_offsets[art_idx] = offsets[idx]
        articles['y_pos'] = [y_offsets[i] for i in articles.index]
        hover_texts       = []
        articles          = articles.sort_values(by = 'x_pos')
        for i, row in articles.iterrows():
            txt          = f"id: {row['id']}<br>"
            meta         = f"{row['author']} ({row['year']}). {row['title']}. {row['journal']}. doi:{row['doi']}"
            wrapped_meta = "<br>".join(textwrap.wrap(meta, width = 50))
            txt          = txt + wrapped_meta
            hover_texts.append(txt)
        node_trace = go.Scatter(
                                x         = articles['x_pos'],
                                y         = articles['y_pos'],
                                mode      = mode,
                                marker    = dict( symbol = 'circle-dot', size = node_size, color = 'blue', line = dict(color = 'rgb(50, 50, 50)', width = 0.15) ),
                                text      = articles['id'] if node_labels else None,
                                hoverinfo = 'text',
                                hovertext = hover_texts,
                                name      = ''
                               )
        Xa, Ya       = [], []
        article_dict = articles.set_index('id').to_dict(orient = 'index')
        for (src, tgt) in citations:
            if (str(src) in article_dict and str(tgt) in article_dict):
                Xa.append(article_dict[str(src)]['x_pos'])
                Ya.append(article_dict[str(src)]['y_pos'])
                Xa.append(article_dict[str(tgt)]['x_pos'])
                Ya.append(article_dict[str(tgt)]['y_pos'])
                Xa.append(None)
                Ya.append(None)
        edge_trace = go.Scatter(
                                x         = Xa, 
                                y         = Ya,
                                mode      = 'lines',
                                line      = dict(color = 'rgba(0, 0, 0, 0.15)', width = 0.5, dash = 'dot'),
                                hoverinfo = 'none',
                                name      = ''
                                )
        if (len(chain) > 0):
            chain          = [str(c) for c in chain]  
            adjacency_list = {}
            for (src, tgt) in citations:
                src_str, tgt_str = str(src), str(tgt)
                adjacency_list.setdefault(src_str, []).append(tgt_str)
            neighbors = set()
            if (path == False):
                for c in chain:
                    if (c in adjacency_list):
                        for neigh in adjacency_list[c]:
                            if (neigh not in chain):
                                neighbors.add(neigh)
            node_colors = []
            for i, row in articles.iterrows():
                art_id_str = str(row['id'])
                if (art_id_str in chain):
                    node_colors.append('green')
                elif (art_id_str in neighbors):
                    node_colors.append('red')
                elif (art_id_str in n_m):
                    node_colors.append('rgba(50, 50, 50, 0.15)')
                else:
                    node_colors.append('rgba(0, 0, 255, 0.15)')
            Xa_chain, Ya_chain = [], []
            Xa_other, Ya_other = [], []
            if (path == False):
                for (src, tgt) in citations:
                    src_str, tgt_str = str(src), str(tgt)
                    if (src_str in article_dict and tgt_str in article_dict):
                        x1 = article_dict[src_str]['x_pos']
                        y1 = article_dict[src_str]['y_pos']
                        x2 = article_dict[tgt_str]['x_pos']
                        y2 = article_dict[tgt_str]['y_pos']
                        if (src_str in chain):
                            Xa_chain.extend([x1, x2, None])
                            Ya_chain.extend([y1, y2, None])
                        else:
                            Xa_other.extend([x1, x2, None])
                            Ya_other.extend([y1, y2, None])
            else:
                for i in range(0, len(chain)-1):
                    src_str, tgt_str = str(chain[i]), str(chain[i+1])
                    if (src_str in article_dict and tgt_str in article_dict):
                        x1 = article_dict[src_str]['x_pos']
                        y1 = article_dict[src_str]['y_pos']
                        x2 = article_dict[tgt_str]['x_pos']
                        y2 = article_dict[tgt_str]['y_pos']
                        Xa_chain.extend([x1, x2, None])
                        Ya_chain.extend([y1, y2, None])
                for (src, tgt) in citations:
                    src_str, tgt_str = str(src), str(tgt)
                    if (src_str in article_dict and tgt_str in article_dict):
                        x1 = article_dict[src_str]['x_pos']
                        y1 = article_dict[src_str]['y_pos']
                        x2 = article_dict[tgt_str]['x_pos']
                        y2 = article_dict[tgt_str]['y_pos']
                        if (src_str not in chain):
                            Xa_other.extend([x1, x2, None])
                            Ya_other.extend([y1, y2, None])

            edge_trace_other = go.Scatter(
                                            x         = Xa_other,
                                            y         = Ya_other,
                                            mode      = 'lines',
                                            line      = dict(color = 'rgba(0, 0, 0, 0.15)', width = 0.5, dash = 'dot'),
                                            hoverinfo = 'none',
                                            name      = ''
                                        )
            edge_trace_chain = go.Scatter(
                                            x         = Xa_chain,
                                            y         = Ya_chain,
                                            mode      = 'lines',
                                            line      = dict(color = 'black', width = 1, dash = 'solid'),
                                            hoverinfo = 'none',
                                            name      = ''
                                        )
            node_trace       = go.Scatter(
                                            x         = articles['x_pos'],
                                            y         = articles['y_pos'],
                                            mode      = mode,
                                            marker    = dict(symbol = 'circle-dot', size = node_size, color = node_colors, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                                            text      = articles['id'] if node_labels else None,
                                            hoverinfo = 'text',
                                            hovertext = hover_texts,
                                            name      = ''
                                        )
            data = [edge_trace_other, edge_trace_chain, node_trace]
        else:
            node_colors = []
            for i, row in articles.iterrows():
                art_id_str = str(row['id'])
                if (art_id_str in n_m):
                    if (min_links > 0):
                        node_colors.append('rgb(50, 50, 50)')
                    else:
                        node_colors.append('blue')
                else:
                    node_colors.append('blue')
            Xa_chain, Ya_chain = [], []
            node_trace = go.Scatter(
                                    x         = articles['x_pos'],
                                    y         = articles['y_pos'],
                                    mode      = mode,
                                    marker    = dict(symbol = 'circle-dot', size = node_size, color = node_colors, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                                    text      = articles['id'] if node_labels else None,
                                    hoverinfo = 'text',
                                    hovertext = hover_texts,
                                    name      = ''
                                )
            data = [edge_trace, node_trace]
        layout = go.Layout( showlegend = False, hovermode = 'closest', margin = dict(b = 10, l = 5, r = 5, t = 10))
        fig    = go.Figure(data = data, layout = layout)
        fig.update_layout(
                            plot_bgcolor = 'rgb(255, 255, 255)',
                            hoverlabel   = dict(font_size = 12),
                            xaxis        = dict(showgrid = False, zeroline = False, showticklabels = True, title = 'Years', tickangle = 90, type = 'category', categoryorder = 'array', categoryarray = years),
                            yaxis        = dict(showgrid = False, zeroline = False, showticklabels = False, title = '')
                        )
        fig.update_traces(textfont_size = font_size, textfont_color = 'yellow')
    
        ymin = min(articles['y_pos'])
        ymax = max(articles['y_pos'])
        pad = dist_pad * 2 if ymax != ymin else dist_pad
        fig.update_yaxes(range = [ymin - pad, ymax + pad])
    
        fig.show()
        self.ask_gpt_hist                        = pd.DataFrame(citations, columns = ['Paper ID', 'Reference ID'])
        self.ask_gpt_hist['Paper ID']            = self.ask_gpt_hist['Paper ID'].astype(str)
        self.ask_gpt_hist                        = self.ask_gpt_hist.merge(articles[['id', 'year']], left_on = 'Paper ID', right_on = 'id', how = 'left')
        self.ask_gpt_hist.rename(columns = {'year': 'Paper_Year'}, inplace = True)
        self.ask_gpt_hist.drop('id', axis = 1, inplace = True)
        self.ask_gpt_hist['Reference ID']        = self.ask_gpt_hist['Reference ID'].astype(str)
        self.ask_gpt_hist                        = self.ask_gpt_hist.merge(articles[['id', 'year']], left_on = 'Reference ID', right_on = 'id', how = 'left')
        self.ask_gpt_hist.rename(columns = {'year': 'Reference_Year'}, inplace = True)
        self.ask_gpt_hist                        = self.ask_gpt_hist[self.ask_gpt_hist['Reference_Year'] <= self.ask_gpt_hist['Paper_Year']]
        condition                                = (self.ask_gpt_hist['Reference_Year'] == self.ask_gpt_hist['Paper_Year'])
        indexes_to_drop                          = []
        for idx, row in self.ask_gpt_hist.loc[condition].iterrows():
            paper_id = int(row['Paper ID'])
            ref_id   = str(row['Reference ID'])
            if (ref_id not in matrix.columns):
                indexes_to_drop.append(idx)
            else:
                if (matrix.loc[paper_id, ref_id] == 0):
                    indexes_to_drop.append(idx)
        self.ask_gpt_hist.drop(index = indexes_to_drop, inplace = True)
        self.ask_gpt_hist['Paper ID (Year)']     = self.ask_gpt_hist.apply(lambda row: f"{row['Paper ID']} ({row['Paper_Year']})", axis = 1)
        self.ask_gpt_hist['Reference ID (Year)'] = self.ask_gpt_hist.apply(lambda row: f"{row['Reference ID']} ({row['Reference_Year']})", axis = 1)
        citations                                = [(int(r), int(c)) for (r, c) in zip(self.ask_gpt_hist['Paper ID'], self.ask_gpt_hist['Reference ID'])]
        self.ask_gpt_hist                        = self.ask_gpt_hist[['Paper ID (Year)', 'Reference ID (Year)']]
        return citations

    # Function: Analyze Citations
    def analyze_hist_citations(self, citations, min_path_size = 2):
        G                     = nx.DiGraph(citations)
        most_referenced_paper = max(G.in_degree,  key = lambda x: x[1], default = (None, 0))
        most_citing_paper     = max(G.out_degree, key = lambda x: x[1], default = (None, 0))
        recent_to_old_paths   = []
        sources               = sorted({src for src, tgt in citations}, reverse = True)
        targets               = sorted({tgt for src, tgt in citations})
        for source in sources:  
            for target in targets:
                if (nx.has_path(G, source, target)):
                    path = nx.shortest_path(G, source = source, target = target)
                    if (len(path) >= min_path_size):
                        recent_to_old_paths.append(path)
        recent_to_old_paths   = sorted(recent_to_old_paths, key = len, reverse = True)
        filtered_paths        = []
        for path in recent_to_old_paths:
            if not any(set(path).issubset(set(existing_path)) for existing_path in filtered_paths):
                filtered_paths.append(path)
        recent_to_old_paths   = filtered_paths
        print('Most Referenced Paper ID:', most_referenced_paper[0], '-> Cited', most_referenced_paper[1], 'Times')
        print('Paper ID that Cites the Most:', most_citing_paper[0], '-> Cites', most_citing_paper[1], 'Papers')
        if (len(recent_to_old_paths ) > 0):
            print('Paper IDs of Longest Citation Path:', recent_to_old_paths[0])
        return recent_to_old_paths 

############################################################################

    # Function: Sentence Embeddings # 'abs', 'title', 'kwa', 'kwp'
    def create_embeddings(self, stop_words = ['en'], rmv_custom_words = [], corpus_type = 'abs', model = 'allenai/scibert_scivocab_uncased'): 
        model = SentenceTransformer(model)
        if  (corpus_type == 'abs'):
            corpus = self.data['abstract']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'title'):
            corpus = self.data['title']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'kwa'): 
            corpus = self.data['author_keywords']
            corpus = corpus.tolist()
        elif (corpus_type == 'kwp'):
            corpus = self.data['keywords']
            corpus = corpus.tolist()
        self.embds = model.encode(corpus)
        return 

############################################################################

    # Function: Topics - Create
    def topics_creation(self, stop_words = ['en'], rmv_custom_words = [], embeddings = False, model = 'allenai/scibert_scivocab_uncased'):
        umap_model = UMAP(n_neighbors = 15, n_components = 5, min_dist = 0.0, metric = 'cosine', random_state = 1001)
        if (embeddings ==  False):
            self.topic_model = BERTopic(umap_model = umap_model, calculate_probabilities = True)
        else:
            sentence_model   = SentenceTransformer(model)
            self.topic_model = BERTopic(umap_model = umap_model, calculate_probabilities = True, embedding_model = sentence_model)
        self.topic_corpus       = self.clear_text(self.data['abstract'], stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words, verbose = False)
        self.topics, self.probs = self.topic_model.fit_transform(self.topic_corpus)
        self.topic_info         = self.topic_model.get_topic_info()
        print(self.topic_info)
        return  
  
    # Function: Topics - Load
    def topics_load_file(self,  saved_file = 'my_topic_model'):
        self.topic_model  = BERTopic.load(saved_file)
        self.topic_corpus = self.clear_text(self.data['abstract'], stop_words = [], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [], verbose = False)
        self.topics       = self.topic_model.topics_               
        self.probs        = self.topic_model.probabilities_ 
        self.topic_info   = self.topic_model.get_topic_info()
        print(self.topic_info)
        return
    
    # Function: Topics - Main Representatives
    def topics_representatives(self):
        docs        = [[] for _ in range(0, self.topic_info.shape[0])]
        papers      = self.topic_model.get_representative_docs()
        self.df_rep = pd.DataFrame(np.zeros((self.topic_info.shape[0], 2)), columns = ['Topic', 'Docs'])
        for i in range(0, self.topic_info.shape[0]):
            if (self.topic_info.iloc[i, 0] != -1):
                paper = papers[self.topic_info.iloc[i, 0]]
                for item in paper:
                    docs[i].append(self.topic_corpus.index(item))
            self.df_rep.iloc[i, 0] = self.topic_info.iloc[i, 0]
            self.df_rep.iloc[i, 1] = '; '.join(map(str, docs[i]))
        return self.df_rep
        
    # Function: Topics - Reduce
    def topics_reduction(self, topicsn = 3):
        self.topics, self.probs = self.topic_model.reduce_topics(docs = self.topic_corpus, nr_topics = topicsn - 1)
        self.topic_info         = self.topic_model.get_topic_info()
        print(self.topic_info)
        return
    
    # Function: Graph Topics - Topics
    def graph_topics(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = ['Topic ' + str(self.topic_info.iloc[i, 0]) + ' ( Count = ' + str(self.topic_info.iloc[i, 1]) + ') ' for i in range(0, self.topic_info.shape[0])]
        column       = 1
        columns      = 4
        row          = 1
        rows         = int(np.ceil(self.topic_info.shape[0] / columns))
        fig          = ps.make_subplots(rows               = rows,
                                        cols               = columns,
                                        shared_yaxes       = False,
                                        shared_xaxes       = False,
                                        horizontal_spacing = 0.1,
                                        vertical_spacing   = 0.4 / rows if rows > 1 else 0,
                                        subplot_titles     = topics_label
                                        )
        for i in range(0, self.topic_info.shape[0]):
            sequence = self.topic_model.get_topic(self.topic_info.iloc[i, 0])
            words    = [str(item[0]) for item in sequence]
            values   = [str(item[1]) for item in sequence]
            trace    = go.Bar(x           = values,
                              y           = words,
                              orientation = 'h',
                              marker      = dict(color = self.color_names[i], line = dict(color = 'black', width = 1))
                              )
            fig.append_trace(trace, row, column)
            if (column == columns):
                column = 1
                row    = row + 1
            else:
                column = column + 1
        fig.update_xaxes(showticklabels = False)
        fig.update_layout(paper_bgcolor = 'rgb(255, 255, 255)', plot_bgcolor = 'rgb(255, 255, 255)', showlegend = False)
        fig.show()
        return 
    
    # Function: Graph Topics - Topics Distribution
    def graph_topics_distribution(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = []
        topics_count = []
        words        = []
        for i in range(0, self.topic_info.shape[0]):
            topics_label.append('Topic ' + str(self.topic_info.iloc[i, 0]))
            topics_count.append(self.topic_info.iloc[i, 1])
            sequence = self.topic_model.get_topic(self.topic_info.iloc[i, 0])
            sequence = ['-'+str(item[0]) for item in sequence]
            words.append('Count: ' + str(self.topic_info.iloc[i, 1]) +'<br>'+'<br>'+ 'Words: ' +'<br>'+ '<br>'.join(sequence))
        fig = go.Figure(go.Bar(x           = topics_label,
                               y           = topics_count,
                               orientation = 'v',
                               hoverinfo   = 'text',
                               hovertext   = words,
                               marker      = dict(color = 'rgba(78, 246, 215, 0.6)', line = dict(color = 'black', width = 1)),
                               name        = ''
                              ),
                        )
        fig.update_xaxes(zeroline = False)
        fig.update_layout(paper_bgcolor = 'rgb(189, 189, 189)', plot_bgcolor = 'rgb(189, 189, 189)')
        fig.show()
        return
    
    # Function: Graph Topics - Projected Topics 
    def graph_topics_projection(self, view = 'browser', method = 'tsvd'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = []
        topics_count = []
        words        = []
        for i in range(0, self.topic_info.shape[0]):
            topics_label.append(str(self.topic_info.iloc[i, 0]))
            topics_count.append(self.topic_info.iloc[i, 1])
            sequence  = self.topic_model.get_topic(self.topic_info.iloc[i, 0])
            sequence  = ['-'+str(item[0]) for item in sequence]
            words.append('Count: ' + str(self.topic_info.iloc[i, 1]) +'<br>'+'<br>'+ 'Words: ' +'<br>'+ '<br>'.join(sequence))
        try:
            embeddings = self.topic_model.c_tf_idf.toarray()
        except:
            embeddings = self.topic_model.c_tf_idf_.toarray()
        if (method.lower() == 'umap'):
            decomposition = UMAP(n_components = 2, random_state = 1001)
        else:
            decomposition = tsvd(n_components = 2, random_state = 1001)
        transformed   = decomposition.fit_transform(embeddings)
        fig           = go.Figure(go.Scatter(x           = transformed[:,0],
                                             y           = transformed[:,1],
                                             opacity     = 0.85,
                                             mode        = 'markers+text',
                                             marker      = dict(symbol = 'circle-dot', color = 'rgba(250, 240, 52, 0.75)', line = dict(color = 'black', width = 1)), 
                                             marker_size = topics_count,
                                             text        = topics_label,
                                             hoverinfo   = 'text',
                                             hovertext   = words,
                                             name        = ''
                                             ),
                                  )
        x_range = (transformed[:,0].min() - abs((transformed[:,0].min()) * .35), transformed[:,0].max() + abs((transformed[:,0].max()) * .35))
        y_range = (transformed[:,1].min() - abs((transformed[:,1].min()) * .35), transformed[:,1].max() + abs((transformed[:,1].max()) * .35))
        fig.update_xaxes(range = x_range, showticklabels = False)
        fig.update_yaxes(range = y_range, showticklabels = False)
        fig.add_shape(type = 'line', x0 = sum(x_range)/2, y0 = y_range[0], x1 = sum(x_range)/2, y1 = y_range[1], line = dict(color = 'rgb(0, 0, 0)', width = 0.5))
        fig.add_shape(type = 'line', x0 = x_range[0], y0 = sum(y_range)/2, x1 = x_range[1], y1 = sum(y_range)/2, line = dict(color = 'rgb(0, 0, 0)', width = 0.5))
        fig.add_annotation(x = x_range[0], y = sum(y_range)/2, text = '<b>D1<b>', showarrow = False, yshift = 10)
        fig.add_annotation(y = y_range[1], x = sum(x_range)/2, text = '<b>D2<b>', showarrow = False, xshift = 10)
        fig.update_layout(paper_bgcolor = 'rgb(235, 235, 235)', plot_bgcolor = 'rgb(235, 235, 235)', xaxis = dict(showgrid = False, zeroline = False), yaxis = dict(showgrid = False, zeroline = False))
        fig.show()
        return 
    
    # Function: Graph Topics - Topics Heatmap
    def graph_topics_heatmap(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = []
        try:
            embeddings = self.topic_model.c_tf_idf.toarray()
        except:
            embeddings = self.topic_model.c_tf_idf_.toarray()
        dist_matrix  = cosine_similarity(embeddings)
        for i in range(0, self.topic_info.shape[0]):
            topics_label.append('Topic ' + str(self.topic_info.iloc[i, 0]))
        trace = go.Heatmap(z          = dist_matrix,
                           x          = topics_label,
                           y          = topics_label,
                           zmin       = -1,
                           zmax       =  1,
                           xgap       =  1,
                           ygap       =  1,
                           text       = np.around(dist_matrix, decimals = 2),
                           hoverinfo  = 'text',
                           colorscale = 'thermal'
                          )
        layout = go.Layout(title_text = 'Topics Heatmap', xaxis_showgrid = False, yaxis_showgrid = False, yaxis_autorange = 'reversed')
        fig    = go.Figure(data = [trace], layout = layout)
        fig.show()
        return
    
    # Function:  Graph Topics - Topics Over Time
    def graph_topics_time(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        idx              = self.data['year'] != -1
        abstract         = self.data['abstract'][idx]
        year             = self.data['year'][idx]
        topics_over_time = self.topic_model.topics_over_time(abstract, year, nr_bins = self.date_end - self.date_str)
        topics           = topics_over_time['Topic'].unique()
        topics           = topics[topics != -1]  
        topics           = np.sort(topics)    
        c                = self.color_names
        fig              = go.Figure()
        for i in range(0, topics.shape[0]):
            topic_data = topics_over_time[topics_over_time['Topic'] == topics[i]]
            fig.add_trace(go.Scatter(
                                     x      = topic_data['Timestamp'],
                                     y      = topic_data['Frequency'],
                                     mode   = 'lines+markers',
                                     name   = f'Topic {topics[i]}',
                                     line   = dict(color = c[i], width = 2.5, shape  = 'spline'),
                                     marker = dict(color = c[i], size = 8,    symbol = 'square')
                                   ))
        fig.update_layout(
                          title        = 'Topic Trends Over Time',
                          xaxis_title  = 'Year',
                          yaxis_title  = 'Frequency',
                          legend_title = 'Topics'
                      )
        fig.show()
        return

    # Function: Topics - Doc Words Distribution   
    def topics_words(self, doc_id = 0):
        abstracts    = self.data['abstract']
        topic, token = self.topic_model.approximate_distribution(abstracts, calculate_tokens = True)
        df           = self.topic_model.visualize_approximate_distribution(abstracts[doc_id], token[doc_id])
        df           = df.data.T
        df.columns   = [f"Topic {col.split('_')[0]}" if col.split('_')[0].isdigit() else col for col in df.columns]
        return df

    # Function: Topics - Topics Collab   
    def topics_authors(self, topn = 15):
        self.__adjacency_matrix_aut(0)
        collab_data        = self.matrix_a.copy(deep = True)
        collab_data        = collab_data.reset_index(drop = True)
        targets            = [item for item in self.u_aut]
        sizes              = self.doc_aut
        idx                = sorted(range(len(sizes)), key = sizes.__getitem__)
        idx.reverse()
        targets            = [targets[i] for i in idx]
        targets            = targets[:topn]
        doc_id             = self.table_id_doc.copy(deep = True)
        doc_id['Document'] = doc_id['Document'].str.lower()
        doc_id['Topics']   = self.topics
        doc_id['Author']   = doc_id['Document'].apply( lambda x: [author for author in targets if author in x])
        doc_id              = doc_id[doc_id['Author'].apply(len) > 0]
        unique_topics       = sorted(set(topic for topic in self.topics))
        summary             = pd.DataFrame(index = targets, columns = unique_topics + ['Total']).fillna(0)
        for author in targets:
            author_docs = doc_id[doc_id['Document'].str.contains(author)]
            for topic in self.topics:
                topic_count               = author_docs[author_docs['Topics'] == topic].shape[0]
                summary.at[author, topic] = topic_count
        summary['Total']   = summary.sum(axis = 1)
        summary            = summary.sort_values(by = 'Total', ascending = False)
        return summary

############################################################################
    
    # Function: W2V
    def word_embeddings(self, stop_words = ['en'], lowercase = True, rmv_accents = True, rmv_special_chars = False, rmv_numbers = True, rmv_custom_words = [], vector_size = 100, window = 5, min_count = 1, epochs = 10):
        
        #----------------------------------------------------------------------
        
        def tokenize(text):
            return re.findall(r'\b\w+\b', text)
        
        #----------------------------------------------------------------------
        
        corpus = self.data['abstract'].tolist()
        corpus = self.clear_text(corpus, 
                                 stop_words        = stop_words, 
                                 lowercase         = lowercase, 
                                 rmv_accents       = rmv_accents, 
                                 rmv_special_chars = rmv_special_chars, 
                                 rmv_numbers       = rmv_numbers, 
                                 rmv_custom_words  = rmv_custom_words, 
                                 verbose           = False)
        corpus = [tokenize(doc) for doc in corpus]
        model  = FastText(sentences   = corpus, 
                          vector_size = vector_size, 
                          window      = window, 
                          min_count   = min_count, 
                          epochs      = epochs)
        
        w_emb  = [model.wv[word] for word in corpus[0] if word in model.wv]
        vocab  = model.wv.index_to_key 
        return model, corpus, w_emb, vocab
    
    # Function: Find Documents that have the Target Words
    def word_embeddings_find_doc(self, corpus, target_words = []):
        results  = []
        i        = -1
        original = self.data['abstract'].tolist()
        for tokens in corpus:
            i = i + 1
            if (set(target_words).issubset(tokens)):
                results.append((i, original[i]))  
        return results
    
    
    # Function: Words Similarity
    def word_embeddings_sim(self, model, word_1 = '', word_2 = ''):
        similarity = model.wv.similarity(word_1 , word_2)
        return similarity
    
    # Function: Words Operations
    def word_embeddings_operations(self, model, positive = [], negative = [], topn = 10):
        result = model.wv.most_similar(positive = positive, negative = negative, topn = topn)
        return result
    
    # Function: Plot Words
    def plot_word_embeddings(self, model, view = 'browser', positive = [], negative = [], topn = 5, node_size = 10, font_size = 14):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        all_words   = []
        all_vectors = []
        all_labels  = []
        for i, (pos, neg) in enumerate(zip(positive, negative)):
            query_label = '+'.join(pos) + ('-' + '-'.join(neg) if neg else '')
            try:
                similar_words = model.wv.most_similar(positive = pos, negative = neg, topn = topn)
                all_words.extend(pos)
                all_vectors.extend([model.wv[word] for word in pos])
                all_labels.extend([query_label] * len(pos))
                for similar_word, _ in similar_words:
                    all_words.append(similar_word)
                    all_vectors.append(model.wv[similar_word])
                    all_labels.append(query_label)
            except KeyError as e:
                print(f"Warning: One or more words not in the model's vocabulary: {e}")
        reducer         = UMAP(n_components = 2, random_state = 42)
        reduced_vectors = reducer.fit_transform(all_vectors)
        df              = pd.DataFrame(reduced_vectors, columns = ['x', 'y'])
        df['word']      = all_words
        df['label']     = all_labels  
        fig             = go.Figure()
        for i, query_label in enumerate(set(all_labels)):
            group_df = df[df['label'] == query_label]
            fig.add_trace(go.Scatter(
                                      x            = group_df['x'],
                                      y            = group_df['y'],
                                      mode         = 'markers+text',
                                      text         = group_df['word'],
                                      textposition = 'top center',
                                      marker       = dict(
                                                           size    = node_size,
                                                           color   = self.color_names[i],
                                                           line    = dict(width = 1, color = 'DarkSlateGrey'),
                                                           opacity = 0.8
                                                         ),
                                    name          = query_label
                                ))
        fig.update_layout(
            hovermode    = 'closest',
            margin       = dict(b = 10, l = 5, r = 5, t = 10),
            plot_bgcolor = '#f5f5f5',
            xaxis        = dict(  showgrid       = True, 
                                  gridcolor      = 'white',
                                  zeroline       = False, 
                                  showticklabels = False, 
                               ),
            yaxis        = dict(  showgrid       = True,  
                                  gridcolor      = 'white',
                                  zeroline       = False, 
                                  showticklabels = False,
                                ),
            legend       = dict( title      = 'Queries',
                                 font       = dict(size = font_size),
                                 itemsizing = 'constant'
                                ),
                        )
        fig.show()
        return

############################################################################

    # Function: Extract Keywords KeyBert or Extract Keywords KeyBert + TextRank
#    def extract_keywords_keybert(self, text, top_n = 10, candidate_factor = 2, stop_words = 'en', rmv_custom_words = [], diversity = True, text_rank = False, ngram = 1, model = 'allenai/scibert_scivocab_uncased'):
#        if not isinstance(text, (list, pd.Series)):
#            corpus = [text]
#        else:
#            corpus = text.tolist()
#        corpus   = self.clear_text(corpus, 
#                                   stop_words        = stop_words, 
#                                   lowercase         = True, 
#                                   rmv_accents       = False, 
#                                   rmv_special_chars = False, 
#                                   rmv_numbers       = False,
#                                   rmv_custom_words  = rmv_custom_words)
#        kw_model  = KeyBERT(model)
#        keywords  = []
#        if text_rank == False:
#            keywords = kw_model.extract_keywords(corpus, keyphrase_ngram_range = (1, ngram), use_maxsum = diversity, top_n = top_n)
#        else:
#            embedds  = SentenceTransformer(model)
#            corpus   = kw_model.extract_keywords(corpus, keyphrase_ngram_range = (1, ngram), use_maxsum = diversity, top_n = top_n * candidate_factor)
#            if isinstance(text, (list, pd.Series)):
#                for c in corpus:
#                    phrases        = [phrase for phrase, _ in c]
#                    embeddings     = embedds.encode(phrases)
#                    sim_matrix     = cosine_similarity(embeddings)
#                    np.fill_diagonal(sim_matrix, 0)
#                    graph          = nx.from_numpy_array(sim_matrix)
#                    scores         = nx.pagerank(graph)
#                    ranked_phrases = sorted(((phrases[i], scores[i]) for i in range(0, len(phrases))), reverse = True)
#                    keywords.append(ranked_phrases[:top_n])
#            else:
#                phrases        = [phrase for phrase, _ in corpus]
#                embeddings     = embedds.encode(phrases)
#                sim_matrix     = cosine_similarity(embeddings)
#                np.fill_diagonal(sim_matrix, 0)
#                graph          = nx.from_numpy_array(sim_matrix)
#                scores         = nx.pagerank(graph)
#                ranked_phrases = sorted(((phrases[i], scores[i]) for i in range(0, len(phrases))), reverse = True)
#                keywords.append(ranked_phrases[:top_n])
#        return keywords
 
############################################################################

    # Function: Abstractive Text Summarization # Model Name List = https://huggingface.co/models?pipeline_tag=summarization&sort=downloads&search=pegasus
    def summarize_abst_peg(self, article_ids = [], model_name = 'google/pegasus-xsum', min_L = 100, max_L = 150):
        abstracts = self.data['abstract']
        corpus    = []
        if (len(article_ids) == 0):
            article_ids = [i for i in range(0, abstracts.shape[0])]
        else:
            article_ids = [int(item) for item in article_ids]
        for i in range(0, abstracts.shape[0]):
            if (abstracts.iloc[i] != 'UNKNOWN' and i in article_ids):
                corpus.append(abstracts.iloc[i])
        if (len(corpus) > 0):
            print('')
            print('Total Number of Valid Abstracts: ', len(corpus))
            print('')
            corpus    = ' '.join(corpus)
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            pegasus   = PegasusForConditionalGeneration.from_pretrained(model_name)
            tokens    = tokenizer.encode(corpus, return_tensors = 'pt', max_length = max_L, truncation = True)
            summary   = pegasus.generate(tokens, min_length = min_L, max_length = max_L, length_penalty = 2.0, num_beams = 4, early_stopping = True)
            summary   = tokenizer.decode(summary[0], skip_special_tokens = True)
        else:
            summary   = 'No abstracts were found in the selected set of documents'
        return summary
    
    # Function: Check Version
    def version_check(self, major, minor, patch):
        try:
            version                   = openai.__version__
            major_v, minor_v, patch_v = [int(v) for v in version.split('.')]
            if ( (major_v, minor_v, patch_v) >= (major, minor, patch) ):
                return True
            else:
                return False
        except AttributeError:
            return False
    
    # Function: Query
    def query_chatgpt(self, prompt, model, max_tokens, n, temperature, flag, api_key):
        if (flag == 0):
          try:
              response = openai.ChatCompletion.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
              response = response['choices'][0]['message']['content']
          except:
              response = openai.Completion.create(engine = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
              response = response.choices[0].text.strip()
        else:
          try:
              client   = openai.OpenAI(api_key = api_key)
              response = client.chat.completions.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
              response = response.choices[0].message.content
          except:
              client   = openai.OpenAI(api_key = api_key)
              response = client.completions.create( model = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
              response = response.choices[0].text.strip()
        return response
            
    # Function: Abstractive Text Summarization
    def summarize_abst_chatgpt(self, article_ids = [], join_articles = False, api_key = 'your_api_key_here', query = 'from the following scientific abstracts, summarize the main information in a single paragraph using around 250 words', model = 'text-davinci-003', max_tokens = 250, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key
        abstracts                = self.data['abstract']
        corpus                   = []
        if (len(article_ids) == 0):
            article_ids = [i for i in range(0, abstracts.shape[0])]
        else:
            article_ids = [int(item) for item in article_ids]
        for i in range(0, abstracts.shape[0]):
            if (abstracts.iloc[i] != 'UNKNOWN' and i in article_ids):
                corpus.append('Abstract (Document ID' + str(i) + '):\n\n')
                corpus.append(abstracts.iloc[i])
                print('Document ID' + str(i) + ' Number of Characters: ' + str(len(abstracts.iloc[i])))
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0
        
        if (len(corpus) > 0):
            print('')
            print('Total Number of Valid Abstracts: ', int(len(corpus)/2))
            print('')
            if (join_articles == False):
                for i, abstract in enumerate(corpus):
                    prompt = query + ':\n\n' + f'{i+1}. {corpus}\n'
            else:    
                corpus = ' '.join(corpus)
                prompt = query + ':\n\n' + f'{i+1}. {corpus}\n'
            summary = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        else:
            summary = 'No abstracts were found in the selected set of documents'
        return summary

    # Function: Abstractive Text Summarization
    def summarize_abst_gemini(self, article_ids = [],join_articles = False, api_key = 'your_api_key_here', query = 'from the following scientific abstracts, summarize the main information in a single paragraph using around 250 words', model_name = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        model     = genai.GenerativeModel(model_name)
        abstracts = self.data['abstract']
        corpus    = []
        if (len(article_ids) == 0):
            article_ids = [i for i in range(0, abstracts.shape[0])]
        else:
            article_ids = [int(item) for item in article_ids]
        for i in range(0, abstracts.shape[0]):
            if (abstracts.iloc[i] != 'UNKNOWN' and i in article_ids):
                corpus.append(abstracts.iloc[i])
        if (len(corpus) > 0):
            print('')
            print('Total Number of Valid Abstracts: ', int(len(corpus)/2))
            print('')
            if (join_articles == False):
                for i, abstract in enumerate(corpus):
                    prompt = query + ':\n\n' + f'{i+1}. {corpus}\n'
            else:    
                corpus = ' '.join(corpus)
                prompt = query + ':\n\n' + f'{i+1}. {corpus}\n'
            summary = model.generate_content(prompt)
            summary = summary .text
        else:
            summary = 'No abstracts were found in the selected set of documents'
        return summary

    # Function: Extractive Text Summarization
    def summarize_ext_bert(self, article_ids = []):
        abstracts = self.data['abstract']
        corpus    = []
        if (len(article_ids) == 0):
            article_ids = [i for i in range(0, abstracts.shape[0])]
        else:
            article_ids = [int(item) for item in article_ids]
        for i in range(0, abstracts.shape[0]):
            if (abstracts.iloc[i] != 'UNKNOWN' and i in article_ids):
                corpus.append(abstracts.iloc[i])
        if (len(corpus) > 0):
            print('')
            print('Total Number of Valid Abstracts: ', len(corpus))
            print('')
            corpus     = ' '.join(corpus)
            bert_model = Summarizer()
            summary    = ''.join(bert_model(corpus, min_length = 5))
        else:
            summary    = 'No abstracts were found in the selected set of documents'
        return summary

############################################################################

    # Function: Ask chatGPT about Productivity by Year
    def ask_chatgpt_ap(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, related to authors productivity by year', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8, entry = 'aut'):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key
        corpus = ''
        df     = []
        if (entry == 'cout'):
            df = self.ask_gpt_cp
        elif (entry == 'inst'):
            df = self.ask_gpt_ip
        elif (entry == 'jou'):
            df = self.ask_gpt_sp
        elif (entry == 'aut'):
            df = self.ask_gpt_ap
        for element, row in df.iterrows():
            years        = [(year, row[year]) for year in row.index if row[year] > 0]
            paper_counts = ', '.join([f'({year}: {count} paper{"s" if count > 1 else ""})' for year, count in years])
            corpus       = corpus +  f'{element} {paper_counts}\n'
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask chatGPT about Bar Plots 
    def ask_chatgpt_bp(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key
        corpus                   = self.ask_gpt_bp.to_string(index = False)    
        prompt                   = query + ' regarding ' + self.ask_gpt_bp_t + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask chatGPT about Citation Analysis 
    def ask_chatgpt_citation(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key
        corpus                   = self.ask_gpt_nad.to_string(index = False)    
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Collaboration Analysis
    def ask_chatgpt_col_an(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following network information, knowing that Node 1 is connected with Node 2', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key        
        corpus                   = self.ask_gpt_adj.to_string(index = False)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about EDA Report 
    def ask_chatgpt_eda(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.ask_gpt_rt.to_string(index = False)    
        lines                    = corpus.split('\n')
        corpus                   = '\n'.join(' '.join(line.split()) for line in lines)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask chatGPT about Evolution Plot
    def ask_chatgpt_ep(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, related to words apperance by year', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.ask_gpt_ep
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Citation Analysis 
    def ask_chatgpt_hist(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information relating the most influential references, also discover if there is relevant network connections', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = []
        for i in range(0, self.ask_gpt_hist.shape[0]):
            corpus.append('Paper ' + self.ask_gpt_hist.iloc[i,0] + ' Cites Paper ' + self.ask_gpt_hist.iloc[i,1])
        corpus         = ', '.join(corpus)    
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Map Analysis 
    def ask_chatgpt_map(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.ask_gpt_map.to_string(index = False)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about N-Grms 
    def ask_chatgpt_ngrams(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information relating the n-grams and their frequency', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.ask_gpt_ng.to_string(index = False)  
        lines                    = corpus.split('\n')
        corpus                   = '\n'.join(' '.join(line.split()) for line in lines)  
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Sankey Diagram
    def ask_chatgpt_sankey(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information from a network called Sankey', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.ask_gpt_sk.to_string(index = False)   
        lines                    = corpus.split('\n')
        corpus                   = '\n'.join(' '.join(line.split()) for line in lines)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Similarity Analysis 
    def ask_chatgpt_sim(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.ask_gpt_sim.to_string(index = False)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Wordcloud 
    def ask_chatgpt_wordcloud(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = pd.DataFrame.from_dict(self.ask_gpt_wd, orient = 'index', columns = ['Frequency'])    
        corpus                   = corpus.reset_index().rename(columns = {'index': 'Word'})
        corpus                   = corpus.to_string(index = False)
        lines                    = corpus.split('\n')
        corpus                   = '\n'.join(' '.join(line.split()) for line in lines)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask chatGPT about Network Collab
    def ask_chatgpt_net_collab(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, the main nodes represent key entities, and the links indicate their direct connections or relationships', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = []
        for entry in self.ask_gpt_ct:
            target = entry[0][0]  
            ct    = ', '.join(entry[1]) 
            corpus.append(f'Main Node = {target}; Links = {ct}')
        corpus                   = '\n'.join(corpus)
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0

        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask chatGPT about Heatmap
    def ask_chatgpt_heat(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, the cell represents the article IDs where the row and column appear together', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
        flag                     = 0
        os.environ['OPENAI_KEY'] = api_key 
        corpus                   = self.heat_y_x.to_string()
        corpus                   = self.heat_y_x.to_csv(sep = "\t")
        prompt                   = query + ':\n\n' + f'{corpus}\n'
        prompt                   = prompt[:char_limit]
        
        if (self.version_check(1, 0, 0)):
            flag = 1
        else:
            flag = 0
            
        analyze = self.query_chatgpt(prompt, model, max_tokens, n, temperature, flag, api_key)
        print('Number of Characters: ' + str(len(prompt)))
        return analyze  
    
############################################################################

    # Function: Ask Gemini about Productivity by Year
    def ask_gemini_ap(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, related to authors productivity by year', model = 'gemini-1.5-flash', entry = 'aut'):
        genai.configure(api_key = api_key)
        gem    = genai.GenerativeModel(model)
        corpus = ''
        df     = []
        if (entry == 'cout'):
            df = self.ask_gpt_cp
        elif (entry == 'inst'):
            df = self.ask_gpt_ip
        elif (entry == 'jou'):
            df = self.ask_gpt_sp
        elif (entry == 'aut'):
            df = self.ask_gpt_ap
        for element, row in df.iterrows():
            years        = [(year, row[year]) for year in row.index if row[year] > 0]
            paper_counts = ', '.join([f'({year}: {count} paper{"s" if count > 1 else ""})' for year, count in years])
            corpus       = corpus +  f'{element} {paper_counts}\n'
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask Gemini about Bar Plots 
    def ask_gemini_bp(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_bp.to_string(index = False)    
        prompt  = query + ' regarding ' + self.ask_gpt_bp_t + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask Gemini about Citation Analysis 
    def ask_gemini_citation(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_nad.to_string(index = False)    
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Collaboration Analysis
    def ask_gemini_col_an(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following network information, knowing that Node 1 is connected with Node 2', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)       
        corpus  = self.ask_gpt_adj.to_string(index = False)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about EDA Report 
    def ask_gemini_eda(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model) 
        corpus  = self.ask_gpt_rt.to_string(index = False)    
        lines   = corpus.split('\n')
        corpus  = '\n'.join(' '.join(line.split()) for line in lines)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask Gemini about Evolution Plot
    def ask_gemini_ep(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, related to words apperance by year', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_ep
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Citation Analysis 
    def ask_gemini_hist(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information relating the most influential references, also discover if there is relevant network connections', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem    = genai.GenerativeModel(model)
        corpus = []
        for i in range(0, self.ask_gpt_hist.shape[0]):
            corpus.append('Paper ' + self.ask_gpt_hist.iloc[i,0] + ' Cites Paper ' + self.ask_gpt_hist.iloc[i,1])
        corpus  = ', '.join(corpus)    
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Map Analysis 
    def ask_gemini_map(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_map.to_string(index = False)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about N-Grms 
    def ask_gemini_ngrams(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information relating the n-grams and their frequency', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_ng.to_string(index = False)  
        lines   = corpus.split('\n')
        corpus  = '\n'.join(' '.join(line.split()) for line in lines)  
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Sankey Diagram
    def ask_gemini_sankey(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information from a network called Sankey', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_sk.to_string(index = False)   
        lines   = corpus.split('\n')
        corpus  = '\n'.join(' '.join(line.split()) for line in lines)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Similarity Analysis 
    def ask_gemini_sim(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.ask_gpt_sim.to_string(index = False)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Wordcloud 
    def ask_gemini_wordcloud(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = pd.DataFrame.from_dict(self.ask_gpt_wd, orient = 'index', columns = ['Frequency'])    
        corpus  = corpus.reset_index().rename(columns = {'index': 'Word'})
        corpus  = corpus.to_string(index = False)
        lines   = corpus.split('\n')
        corpus  = '\n'.join(' '.join(line.split()) for line in lines)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze
    
    # Function: Ask Gemini about Network Collab
    def ask_gemini_net_collab(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, the main nodes represent key entities, and the links indicate their direct connections or relationships', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem    = genai.GenerativeModel(model)
        corpus = []
        for entry in self.ask_gpt_ct:
            target = entry[0][0]  
            ct    = ', '.join(entry[1]) 
            corpus.append(f'Main Node = {target}; Links = {ct}')
        corpus  = '\n'.join(corpus)
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze

    # Function: Ask Gemini about Heatmap
    def ask_gemini_heat(self, char_limit = 4097, api_key = 'your_api_key_here', query = 'give me insights about the following information, the cell represents the article IDs where the row and column appear together', model = 'gemini-1.5-flash'):
        genai.configure(api_key = api_key)
        gem     = genai.GenerativeModel(model)
        corpus  = self.heat_y_x.to_string()
        corpus  = self.heat_y_x.to_csv(sep = "\t")
        prompt  = query + ':\n\n' + f'{corpus}\n'
        prompt  = prompt[:char_limit]
        analyze = gem.generate_content(prompt)
        analyze = analyze.text
        print('Number of Characters: ' + str(len(prompt)))
        return analyze    

############################################################################
