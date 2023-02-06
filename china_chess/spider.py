import requests
import re
import time

def findstr(pattern, string):
    # 正则表达式匹配字符串
    ans = re.search(pattern, string)
    if ans:
        span = ans.span()
        return string[span[0] : span[1]]
    return None

if __name__ == "__main__":
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 Edg/109.0.1518.78"
    }
    url = "https://www.xqbase.com/xqbase/?gameid="
    ans = []
    for i in range(1, 12142):
        try:
            page = requests.get(url = url + str(i), headers = head).text
            page = re.sub(r'\s', '', page)
            res = findstr(r'(?<=pre>)(.*?)(?=</pre)', page)
            ans.append(res + "\n")
        except Exception as e:
            continue
        print(i)
    with open("res.txt", "w+") as f:
        f.writelines(ans)
