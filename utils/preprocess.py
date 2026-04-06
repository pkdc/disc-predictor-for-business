import re
import quopri
from bs4 import BeautifulSoup

email_endings = ['best regards', 'regards', 'all the best', 'sincerely', 'cheers',
                'thank you', 'warm regards', 'yours truly', 'yours sincerely', 'yours faithfully',
                'yours', 'kind regards', 'cordially', 'respectfully', 'with regards', 'with best regards',
                'with warm regards', 'with sincere regards', 'with gratitude', 'with appreciation', 'with thanks',
                'with warmest regards', 'with deepest gratitude', 'with deepest appreciation', 'with deepest thanks',
                'thanks and regards', 'many thanks']
escaped_endings = [re.escape(endings) for endings in email_endings]

common_signature = ['Enron North America Corp.', 'Carol St. Clair', 'Debra Perlingiere', 'Keegan Farrell', 'Eric Bass']
escaped_signature = [re.escape(sig) for sig in common_signature]


IMAGE_PATTERN = re.compile(r'\[IMAGE\]')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
FILE_PATTERN = re.compile(r'[-|\s]?[a-zA-Z0-9_-]+\.(?:txt|pdf|docx?|xlsx?|pptx?|csv)')
PHONE_NUM_START_PATTERN = re.compile(r'(?im)^\s*(?:\(?(phone|fax|mobile|mob|tel|office)\)?:?\s*)?(?:\+?\d{1,3}[\s\-]?)?\d{2,4}[\s\-]?\d{3}[\s\-]?\d{4}\s*$')
EMAIL_SIGNOFF_PATTERN = re.compile(r'(?im)^\s*(?:' + "|".join(escaped_endings) + r')\s*[,.]?\s*$.*', re.DOTALL)
EMAIL_SIGNATURE_PATTERN = re.compile(r'(?im)^\s*(?:' + "|".join(escaped_signature) + r')[\s\S]*')
EMAIL_HEADER_PATTERN = re.compile(r'(?:Subject|To|Cc|Bcc|From|Sent|Content-Type|Content-Transfer-Encoding|MIME-Version):\s+.*\n', re.IGNORECASE)
PRICE_MASK_PATTERN = re.compile(r'\$\d+(?:\.\d{1,2})?(?:\s*[-\u2013*]\s*\$\d+(?:\.\d{1,2})?)?')
PERCENTAGE_MASK_PATTERN = re.compile(r'(?:\d+(?:\.\d{1,3})?(?:\s*\%(?:\s*[-\u2013*]\s*\d+(?:\.\d{1,3})?\s*\%?)?))')
PHONE_PATTERN = re.compile(r'(?i)(?:\+?\d{1,3}[\s\-\.]?)?\(?\d{2,4}\)?[\s\-\.]?\d{2,3}[\s\-\.]?\d{3,4}')
LOCAL_PHONE_PATTERN = re.compile(r'\b\d{3}[-\s\.]?\d{4}\b')
EMAIL_PATTERN = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
NUM_DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b')
TEXTUAL_DATE_PATTERN = re.compile(r'(?i)\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4})\b')
TIME_PATTERN = re.compile(r'(?i)\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\b')
RECIPIENT_BLOCK_PATTERN = re.compile(r'(?i)\n\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?/[A-Z]+@[\w/]+[\s\S]*',)
TAB_SPACE_PATTERN = re.compile(r'[ \t]+')
MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n{2,}')
# PROMO_PATTERN = re.compile(r"""
#                         unsubscribe|limited time offer|buy now|discount|promo code|click here|order now|shop now
#                         |save now|exclusive deal|offer expires|last chance|order today|shop today|limited offer
#                         |special offer|guarantee"""
#                         ,re.IGNORECASE | re.VERBOSE)

def clean_html(msg_body):
    return BeautifulSoup(msg_body, 'html.parser').get_text()

def clean_msg_body(msg_body):
    if not isinstance(msg_body, str):
        return ''

    # Try decoding Quoted-Printable format
    try:
        msg_body = quopri.decodestring(msg_body).decode('utf-8', errors='ignore')
    except Exception:
        pass   # If decoding fails, keep the original message

    msg_body = clean_html(msg_body)
    msg_body = EMAIL_SIGNOFF_PATTERN.sub('', msg_body)
    msg_body = EMAIL_SIGNATURE_PATTERN.sub('', msg_body)
    msg_body = PHONE_NUM_START_PATTERN.sub('', msg_body)
    msg_body = EMAIL_HEADER_PATTERN.sub('', msg_body)
    msg_body = IMAGE_PATTERN.sub('', msg_body)
    msg_body = URL_PATTERN.sub('', msg_body)
    msg_body = FILE_PATTERN.sub('', msg_body)
    msg_body = RECIPIENT_BLOCK_PATTERN.sub('', msg_body)
    msg_body = PRICE_MASK_PATTERN.sub('$X', msg_body)
    msg_body = EMAIL_PATTERN.sub('[EMAIL]', msg_body)
    msg_body = PERCENTAGE_MASK_PATTERN.sub('X%', msg_body)
    msg_body = PHONE_PATTERN.sub('[PHONE]', msg_body)
    msg_body = LOCAL_PHONE_PATTERN.sub('[PHONE]', msg_body)
    msg_body = NUM_DATE_PATTERN.sub('[DATE]', msg_body)
    msg_body = TEXTUAL_DATE_PATTERN.sub('[DATE]', msg_body)
    msg_body = TIME_PATTERN.sub('[TIME]', msg_body)
    msg_body = TAB_SPACE_PATTERN.sub(' ', msg_body)

    # Collapse multiple newlines, but preserve one newline between blocks of text
    msg_body = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', msg_body)

    # Strip each line and remove lines that are purely whitespace
    lines = [line.strip() for line in msg_body.split('\n')]
    msg_body = '\n'.join([line for line in lines if line])

    msg_body = msg_body.strip()

    return msg_body