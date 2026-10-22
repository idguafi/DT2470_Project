import csv
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd


# Vad är mitt mål här?
# Mänsklig data + AI data i ETT dataset.
# Vill ha: lika många från varje set för alla genres.
# Måste:
# * Först extrahera mänsklig metadata -> mappa till relevanta filer
# * Extrahera FULLY ai låtar till disk genom att köra:     for archive in sorted((root / "fake_songs").glob("*.zip")):
#   with zipfile.ZipFile(archive) as source:
#      for member in source.namelist():
# Sedan per rimlig heuristik välja något sätt att få lika många mänskliga och ai låtar per genre.
# Typ minimum av respektive mängds kardinalitet.

# n_genre = min(antal_mänskliga_i_genren, antal_ai_i_genren)


def ExtractArtificialMetadata(directory):
    artifical = []
    with open(directory, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row['label'] == "full fake":
                row['duration'] = float(row['duration'])
                artifical.append(row)
    return pd.DataFrame(artifical)


def ExtractHumanMetadata(directory):
    records = []

    with open(directory, newline="", encoding="utf-8") as file:
        for row in csv.reader(file, delimiter="\t"):
            records.append({
                "track_id": row[0],
                "artist_id": row[1],
                "album_id": row[2],
                "path": row[3],
                "duration": float(row[4]),
                "genres": [
                    tag.removeprefix("genre---")
                    for tag in row[5:]
                    if tag.startswith("genre---")
                ],
            })

    return pd.DataFrame(records)


def GetIntersectionOfGenres(humanGenres, artificialGenres):
    # The aim of this function is to retrieve all shared genres
    unique = set()
    for genres in humanGenres:
        for genre in genres:
            unique.add(genre)
    intersection = unique.intersection(set(artificialGenres))
    return intersection


def GetHumanCandidatesForGenre(humanData, genre):
    # The aim of this function is to get all potential human rows for this genre
    mask = humanData['genres'].apply(lambda genres: genre in genres)
    return humanData.loc[mask].copy()


def GetArtificialCandidates(artificialData, genre):
    # The aim of this function is to get all potential AI rows for this genre
    mask = artificialData['genre'] == genre
    return artificialData.loc[mask].copy()


# NGL this is the longest function name ever don't judge! ahhahahaha
def CountFloorForNumberOfFilesAndGetArtificialAndHumanDataPairsPerGenre(humanCandidates, artificialCandidates, genre):
    # Aim of the function is to see which has fewer files for this genre, human or AI? so we can get even splits across genres.
    n = min(len(humanCandidates), len(artificialCandidates))
    return {genre: (humanCandidates.sample(n=n, random_state=42), artificialCandidates.sample(n=n, random_state=42))}


def GetGenreFiles(genre_intersection):
    pass


def NormalizeAudio(genre_audio_files):
    pass

# The aim of this method is to normalize genre names because they vary between datasets


def NormalizeGenre(genre):
    return re.sub(r"[^a-z0-9]", "", genre.lower().replace("&", "n"))


def IndexArtificialAudio(root):
    # Point of this is to go through the zip files without
    # loading them to disk and checking for files
    index = {}
    for archive in sorted((root / "fake_songs").glob("*.zip")):
        with zipfile.ZipFile(archive) as source:
            for member in source.namelist():
                if member.lower().endswith(".mp3"):
                    filename = Path(member).stem
                    if filename in index:
                        raise ValueError(f"Duplicate AI filename: {filename}")
                    index[filename] = (archive, member)

    return index


def GetClipsFromAudioFiles(directory="tracks.csv", clip_length=45000):
    # This standardizes the files and ensures that all of them have the same length
    # Clip length in ms.
    output = Path("dataset_clips")
    output.mkdir(exist_ok=False)  # Prevent accidentally overwriting a dataset.
    (output / "audio").mkdir()
    source = Path(directory).parent

    sr = 16000
    length_in_seconds = clip_length/1000  # ms to seconds
    frames = int(length_in_seconds*sr)

    records = []

    with open(directory, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            input_path = source/row["audio_path"]
            relative_path = Path("audio") / f"{row["track_id"]}.wav"
            output_path = output/relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg",
                "-n",
                "-i", str(input_path),
                "-t", "45",                # 45 seconds
                "-vn",
                "-ar", "16000",             # sr
                "-ac", "1",                # keep mono
                "-c:a", "pcm_s16le",
                str(output_path),          # .wav extension
            ], check=True)

            row["audio_path"] = relative_path.as_posix()
            records.append(row)

    new_csv = pd.DataFrame(records)
    new_csv.to_csv(output/"tracks.csv", index=False)
    return new_csv


if __name__ == "__main__":

    humanmetadata = ExtractHumanMetadata(
        "data_human/song_describer_14_04_23.mtg-jamendo.tsv")

    artifical = ExtractArtificialMetadata("data_ai/fake_songs.csv")

    humanmetadata['genres'] = humanmetadata['genres'].apply(
        lambda genres: [NormalizeGenre(genre) for genre in genres]
    )

    artifical['genre'] = artifical['genre'].apply(NormalizeGenre)

    shared_genres = GetIntersectionOfGenres(
        humanGenres=humanmetadata['genres'], artificialGenres=artifical['genre'])

    genre_pairs = []
    used_human_ids = set()

    for genre in sorted(shared_genres):
        human_candidate = GetHumanCandidatesForGenre(humanmetadata, genre)
        artificial_candidate = GetArtificialCandidates(artifical, genre)

        human_candidate = human_candidate[
            ~human_candidate["track_id"].isin(used_human_ids)
        ]

        pair = CountFloorForNumberOfFilesAndGetArtificialAndHumanDataPairsPerGenre(
            human_candidate, artificial_candidate, genre
        )

        selected_humans, selected_ais = pair[genre]
        used_human_ids.update(selected_humans["track_id"])
        genre_pairs.append(pair)

    ai_index = IndexArtificialAudio(Path("data_ai"))
    human_root = Path("data_human/audio")

    # Need to check if the files indeed do exist for our genres
    for pair in genre_pairs:
        for genre, (humans, ais) in pair.items():

            for path in humans["path"]:
                audio_path = human_root / Path(path).with_suffix(".2min.mp3")
                if not audio_path.is_file():
                    print(f"Missing human audio: {audio_path}")

            for filename in ais["filename"]:
                if filename not in ai_index:
                    print(f"Missing AI audio: {filename}")

    # If nothing is missing, we continue to actually load these, and store in csv.

    new_data = GetClipsFromAudioFiles()

    new_data.head()
