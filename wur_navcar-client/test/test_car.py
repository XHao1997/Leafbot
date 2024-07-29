import requests
import argparse


def main():

    ip_addr = '192.168.101.19'

    try:
        while True:
            command = input("input your json cmd: ")
            url = "http://" + ip_addr + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            print(content)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()