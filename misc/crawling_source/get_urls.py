from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import requests
import re
import os
import time

# Custom expected condition to check if the element is not None
def element_is_not_none(element):
    return element if element.get_attribute('src') is not None else False

# Set up Chrome options to run in headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode

# Set up the Selenium WebDriver
service = Service('/Users/vinbrain/Downloads/chromedriver-mac-arm64 2/chromedriver')  # Adjust the path to your chromedriver
driver = webdriver.Chrome(service=service, options=chrome_options)

def get_article_urls_dynamic(page_number, type):
    url_list = []
    article_links = set()
    
    url = f'https://tuoitre.vn/{type}/trang-{page_number}.htm'
    driver.get(url)

    SCROLL_PAUSE_TIME = .5  # Time to wait for content to load after scroll
    MAX_ATTEMPTS = 10000  # Limit to prevent endless loops

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        print(f'Current page: {driver.current_url[19:]}')
        url_list.append(driver.current_url)

        print(f'Attempt: {attempt}')
        attempt += 1

        print('Scrolling down...')
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # time.sleep(SCROLL_PAUSE_TIME)
        # Wait until the URL changes
        try:
            WebDriverWait(driver, 1).until(EC.url_changes(driver.current_url))            
        except TimeoutException:
            view_more_button = driver.find_element(By.CLASS_NAME, "view-more")
            print("Same height as previous, need to click 'Xem them' button")
            if view_more_button.is_displayed():
                print('Clicking...')
                # view_more_button.click()
                ActionChains(driver).move_to_element(view_more_button).click(view_more_button).perform()
                
            time.sleep(SCROLL_PAUSE_TIME)


        ##### Get all article elements and save to file #####
        article_elements = driver.find_elements(By.XPATH, '//a[contains(@class, "box-category-link-title")]')
        print(f'Number of articles: {len(article_elements)}')

        error_elements = 0       
        duplicated_articles = 0
        with open(f'./{type}_article_urls.txt', 'a') as f:
            for element in article_elements:
                try:
                    # Wait for the 'href' attribute to be present and not empty
                    href_value = WebDriverWait(driver, 100).until(
                        lambda driver: element.get_attribute('href') if element.get_attribute('href') else False
                    )
                    # element = WebDriverWait(driver, 100).util(lambda driver: element_is_not_none(element))
                    # href_value = element.get_attribute('href')
                    if href_value not in article_links:
                        f.write(f'{href_value}\n')
                        article_links.add(href_value)
                    else:
                        duplicated_articles += 1
                except Exception as e:
                    error_elements += 1
                    continue
        print(f'Number of error articles: {error_elements}')
        print(f'Number of duplicated articles: {duplicated_articles}')
        
        ### Refresh the page to avoid scrolling too much 
        if attempt == 90:
            driver.refresh()
            time.sleep(2)

        print('*' * 100)

        ### Break the loop 
        if driver.current_url == 'https://tuoitre.vn/{type}':
            print('Back to the first site, break.')
            break
            

    print(f'Last page: {url_list[-2]}')

    return article_links

# Get URLs from page 1
article_urls_dynamic = get_article_urls_dynamic(1, type='cong-nghe')

# with open('./article_urls.txt', 'w') as f:
#     for url in article_urls_dynamic:
#         f.write(f'{url}\n')

print(len(article_urls_dynamic))
print(driver.current_url)

# Don't forget to close the browser after you're done
driver.quit()