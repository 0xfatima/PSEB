
import random
import requests


def get_github_profiles(count=3):
    random_start_id = random.randint(1, 100)

    url = "https://api.github.com/users"
    headers = {"Accept": "application/vnd.github+json"}

    params = {
        "since": random_start_id,
        "per_page": count
    }

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            users = response.json()

            for user in users:
                username = user["login"]

                profile_url = f"https://api.github.com/users/{username}"

                profile_response = requests.get(
                    profile_url,
                    headers=headers
                )

                if profile_response.status_code == 200:
                    profile_data = profile_response.json()

                    print("Username:", username)
                    print("Public repos:", profile_data["public_repos"])

                else:
                    print(f"Could not fetch profile for {username}")

        elif response.status_code == 403:
            print("Rate limit reached!")

        else:
            print(f"Error {response.status_code}: Could not fetch users.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


get_github_profiles(3)
