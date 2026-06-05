# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests

def get_exchange_rate(currency_from, currency_to):
    """
    Fetches the current exchange rate between two currencies using the Frankfurter API.

    Args:
        currency_from (str): The ISO 4217 code for the source currency (e.g., 'USD').
        currency_to (str): The ISO 4217 code for the target currency (e.g., 'EUR').

    Returns:
        dict: A dictionary containing the exchange rate data on success, or an error message if the request fails.
              Example success: {"amount": 1.0, "base": "USD", "date": "2023-10-27", "rates": {"EUR": 0.95}}
              Example failure: {"error": "..."}

    Raises:
        requests.exceptions.RequestException: If the network request fails (handled by try-except).
    """
    try:
        url = f"https://api.frankfurter.app/latest"
        params = {
            "from": currency_from,
            "to": currency_to
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}