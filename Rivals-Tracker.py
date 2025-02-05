import streamlit as st
import requests
import cv2
import numpy as np
import pandas as pd
from mss import mss
from PIL import Image
import time
from paddleocr import PaddleOCR
from config import API_KEY

ocr = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)

BASE_URL = 'https://marvelrivalsapi.com/api/v1'
headers = {'x-api-key': API_KEY}

st.title("Rivals Lobby Tracker")

PLAYER_COORDINATES = [
    (267, 309, 813, 372),  
    (267, 381, 813, 444),  
    (267, 456, 813, 519),  
    (269, 530, 815, 593),  
    (269, 604, 815, 667),  
    (264, 678, 810, 741),  
    (1121, 309, 1667, 372), 
    (1121, 381, 1667, 444), 
    (1121, 455, 1667, 518), 
    (1119, 530, 1665, 593), 
    (1119, 603, 1665, 666), 
    (1119, 677, 1665, 740)  
]

def is_team_selection_screen():
    with mss() as sct:
        team_headers = [
            {"top": 185, "left": 135, "width": 140, "height": 40},
            {"top": 185, "left": 1645, "width": 140, "height": 40}
        ]
        
        for pos in team_headers:
            screenshot = sct.grab(pos)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            white_pixels = cv2.countNonZero(cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1])
            
            if white_pixels < 100:
                return False
        return True

def capture_player_names():
    if not is_team_selection_screen():
        return []

    detected_names = []
    with mss() as sct:
        for (x1, y1, x2, y2) in PLAYER_COORDINATES:
            region = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
            screenshot = sct.grab(region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            result = ocr.ocr(opencv_image, cls=True)
            extracted_texts = []
            if result and isinstance(result, list):
                for line in result:
                    if line:
                        for word in line:
                            detected_text = word[1][0].strip()
                            if detected_text and len(detected_text) > 1:
                                extracted_texts.append(detected_text)

            full_name = " ".join(extracted_texts).strip()
            detected_names.append(full_name if full_name else "Unknown")

    while len(detected_names) < 12:
        detected_names.append("Unknown")

    return detected_names[:12]

def fetch_player_stats(player_name):
    response = requests.get(f"{BASE_URL}/player/{player_name}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        heroes_data = data.get("heroes_ranked", [])
        if heroes_data:
            heroes_data.sort(key=lambda x: convert_playtime_to_seconds(x.get("play_time", "0")), reverse=True)
            return heroes_data[:5]
    return None

def convert_playtime_to_seconds(play_time):
    total_seconds = 0
    if 'h' in play_time:
        parts = play_time.split('h')
        hours = int(parts[0])
        total_seconds += hours * 3600
        play_time = parts[1].strip()
    if 'm' in play_time:
        parts = play_time.split('m')
        minutes = int(parts[0])
        total_seconds += minutes * 60
        play_time = parts[1].strip()
    if 's' in play_time:
        parts = play_time.split('s')
        seconds = int(parts[0])
        total_seconds += seconds
    return total_seconds

def calculate_winrate(wins, matches):
    if matches == 0:
        return "0%"
    return f"{round((wins / matches) * 100)}%"

def calculate_average_kda(kills, deaths, assists, matches):
    if matches == 0:
        return "0/0/0"
    avg_kills = round(kills / matches, 1)
    avg_deaths = round(deaths / matches, 1)
    avg_assists = round(assists / matches, 1)
    return f"{avg_kills}/{avg_deaths}/{avg_assists}"

def display_hero_stats(hero_stats):
    import pandas as pd

    rows = []
    for hero in hero_stats:
        name = hero.get("hero_name", "Unknown").title()
        matches = hero.get('matches', 0)
        wins = hero.get('wins', 0)
        play_time = hero.get('play_time', '0')
        win_rate = calculate_winrate(wins, matches)
        avg_kda = calculate_average_kda(
            hero.get('kills', 0),
            hero.get('deaths', 0),
            hero.get('assists', 0),
            matches
        )

        rows.append({
            "Hero": name,
            "Matches": matches,
            "Playtime": play_time,
            "Wins": wins,
            "Winrate": win_rate,
            "Avg KDA": avg_kda
        })

    df = pd.DataFrame(rows)
    df['SortTime'] = df['Playtime'].apply(convert_playtime_to_seconds)
    df.sort_values(by='SortTime', ascending=False, inplace=True)
    df.drop(columns='SortTime', inplace=True)

    st.table(df)

status_placeholder = st.empty()
progress_placeholder = st.empty()

if st.button("Scan Players"):
    status_placeholder.text("Scanning for players...")
    found_players = []
    start_time = time.time()
    duration = 5
    screen_found = False

    try:
        while time.time() - start_time < duration:
            progress = (time.time() - start_time) / duration
            progress_placeholder.progress(progress)

            if not is_team_selection_screen():
                status_placeholder.text(
                    f"Scanning... {int(progress*100)}% (Waiting for team selection screen)"
                )
                time.sleep(0.5)
                continue
            else:
                screen_found = True
                status_placeholder.text(
                    f"Scanning... {int(progress*100)}% (Team selection screen detected)"
                )
                found_players = capture_player_names()
                break  

        progress_placeholder.progress(1.0)

        if screen_found:
            status_placeholder.text("Scan complete!")
        else:
            status_placeholder.text("Scan complete - Team selection screen was not detected")

        if found_players:
            st.header("Your Team")
            for i in range(6):
                st.write(f"**Player {i+1}:** {found_players[i]}")

            st.header("Enemy Team")
            for i in range(6, 12):
                st.write(f"**Player {i+1}:** {found_players[i]}")

            st.header("Player Stats (Top 5 Heroes Each)")
            for player in found_players:
                if player != "Unknown":
                    st.subheader(player)
                    hero_stats = fetch_player_stats(player)
                    if hero_stats:
                        display_hero_stats(hero_stats)  
                    else:
                        st.warning(f"{player} profile is private")
        else:
            st.warning("No player names were detected.")
    except Exception as e:
        st.error(f"Error during scanning: {str(e)}")
    finally:
        progress_placeholder.empty()

username = st.text_input("Enter player username manually:")
if st.button("Search Player"):
    if username:
        st.header(f"Stats for {username}")
        hero_stats = fetch_player_stats(username)
        if hero_stats:
            display_hero_stats(hero_stats)
        else:
            st.error(f"Error: Could not find player '{username}'")
    else:
        st.warning("Please enter a username.")
