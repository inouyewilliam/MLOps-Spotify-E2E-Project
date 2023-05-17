#Import libraries
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#Autentification
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
#with open("C:/Users/willi/Python/Spotify_Project/secret.txt") as f:
    #secret_ls = f.readlines()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


#Playlists Link
#2005
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DWWzQTBs5BHX9?si=0b8e18d6445e44a0"
playlist_2005 = playlist_link.split("/")[-1].split("?")[0]

#2006
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX1vSJnMeoy3V?si=b7943cf5fab5499b"
playlist_2006 = playlist_link.split("/")[-1].split("?")[0]

#2007
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX3j9EYdzv2N9?si=53ae42c9ceeb430c"
playlist_2007 = playlist_link.split("/")[-1].split("?")[0]

#2008
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DWYuGZUE4XQXm?si=f61deefc66e64731"
playlist_2008 = playlist_link.split("/")[-1].split("?")[0]

#2009
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX4UkKv8ED8jp?si=ff5b01a80f3c4015"
playlist_2009 = playlist_link.split("/")[-1].split("?")[0]

#2010
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DXc6IFF23C9jj?si=6fa15385caaa440a"
playlist_2010 = playlist_link.split("/")[-1].split("?")[0]

#2011
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DXcagnSNtrGuJ?si=8b38eb24151a4d98"
playlist_2011 = playlist_link.split("/")[-1].split("?")[0]

#2012
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX0yEZaMOXna3?si=8d852392b8254f93"
playlist_2012 = playlist_link.split("/")[-1].split("?")[0]

#2013
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX3Sp0P28SIer?si=136ddbd666ab4c89"
playlist_2013 = playlist_link.split("/")[-1].split("?")[0]

#2014
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX0h0QnLkMBl4?si=9f02ec79369a4a0c"
playlist_2014 = playlist_link.split("/")[-1].split("?")[0]

#2015
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX9ukdrXQLJGZ?si=2f8192fff8944999"
playlist_2015 = playlist_link.split("/")[-1].split("?")[0]

#2016
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DX8XZ6AUo9R4R?si=691252a9e3e544bd"
playlist_2016 = playlist_link.split("/")[-1].split("?")[0]

#2017
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DWTE7dVUebpUW?si=eaf56e3029d64e0a"
playlist_2017 = playlist_link.split("/")[-1].split("?")[0]

#2018
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DXe2bobNYDtW8?si=35188542f077419f"
playlist_2018 = playlist_link.split("/")[-1].split("?")[0]

#2019
playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DWVRSukIED0e9?si=42c426f041bf4af5"
playlist_2019 = playlist_link.split("/")[-1].split("?")[0]

#2020
playlist_link = "https://open.spotify.com/playlist/2fmTTbBkXi8pewbUvG3CeZ?si=285bf773dc114cb8"
playlist_2020 = playlist_link.split("/")[-1].split("?")[0]

#2021
playlist_link = "https://open.spotify.com/playlist/5GhQiRkGuqzpWZSE7OU4Se?si=02f0d20c63e349bd"
playlist_2021 = playlist_link.split("/")[-1].split("?")[0]

#2022
playlist_link = "https://open.spotify.com/playlist/56r5qRUv3jSxADdmBkhcz7?si=12f218d444ba44fe"
playlist_2022 = playlist_link.split("/")[-1].split("?")[0] 

# Make DataFrame of a playlist
def analyze_playlist(creator, playlist_id):

    #Step 1

    playlist_features_list = ["artist","album","track_name","release_date","popularity","genres","explicit","followers","track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    #Step 2
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["release_date"] = track["track"]["album"]["release_date"]
        playlist_features["popularity"] = track["track"]["popularity"]
        playlist_features["genres"] = [sp.artist(track["track"]["artists"][0]["uri"])["genres"]]
        playlist_features["explicit"] = track["track"]["explicit"]
        playlist_features["followers"] = sp.artist(track["track"]["artists"][0]["uri"])["followers"]["total"]
        playlist_features["track_id"] = track["track"]["id"]
        
        
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[9:]: 
                     (playlist_features[feature]) = (audio_features[feature])
                      
                    
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
    
    #Step 3
        
    return playlist_df

# Make DataFrame of multiple playlists
playlist_dict = {
    "playlist_2005" : ("sp", playlist_2005),
    "playlist_2006" : ("sp", playlist_2006),
    "playlist_2007" : ("sp", playlist_2007),
    "playlist_2008" : ("sp", playlist_2008),
    "playlist_2009" : ("sp", playlist_2009),
    "playlist_2010" : ("sp", playlist_2010),  
    "playlist_2011" : ("sp", playlist_2011), 
    "playlist_2012" : ("sp", playlist_2012),
    "playlist_2013" : ("sp", playlist_2013),
    "playlist_2014" : ("sp", playlist_2014),
    "playlist_2015" : ("sp", playlist_2015),
    "playlist_2016" : ("sp", playlist_2016),
    "playlist_2017" : ("sp", playlist_2017),
    "playlist_2018" : ("sp", playlist_2018),
    "playlist_2019" : ("sp", playlist_2019),
    "playlist_2020" : ("sp", playlist_2020),
    "playlist_2021" : ("sp", playlist_2021),
    "playlist_2022" : ("sp", playlist_2022)
}

def analyze_playlist_dict(playlist_dict):
    
    # Loop through every playlist in the dict and analyze it
    for i, (key, val) in enumerate(playlist_dict.items()):
        playlist_df = analyze_playlist(*val)
        # Add a playlist column so that we can see which playlist a track belongs too
        playlist_df["playlist"] = key
        # Create or concat df
        if i == 0:
            playlist_dict_df = playlist_df
        else:
            playlist_dict_df = pd.concat([playlist_dict_df, playlist_df], ignore_index = True)
            
    return playlist_dict_df

Raw_data_df = analyze_playlist_dict(playlist_dict)

# Clean genres column
df  = pd.DataFrame(Raw_data_df.genres.apply(lambda x: pd.Series(str(x).split("["))))
df2 = df.replace(regex=["]"],value='')
df3 = df2.replace(regex=["'"],value='')
df4 = pd.DataFrame(df3.loc[:,1].apply(lambda x: pd.Series(str(x).split(","))))

# Add Genres columns
Raw_data_df["genres"] = df4.loc[:,0]
Raw_data_df["sub-genres"] = df4.loc[:,1]


# Export to csv
Raw_data_df.to_csv("C:/Users/willi/Python/Spotify_Project/Data/raw_data.csv", index = False)