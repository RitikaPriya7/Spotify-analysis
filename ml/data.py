from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple
from sklearn.preprocessing import OneHotEncoder


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AUDIO_FEATURES = [
    "Danceability",
    "Energy",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Valence",
]

ARTIST_FEATURES = [
    "artist_count",
    "nationality_count",
    # "points_total",
    # "points_individual",
]

TEMPORAL_FEATURES = [
    "year",
    "month",
    "month_sin",
    "month_cos",
    "top10_season_index",
    "season_Spring",      
    "season_Summer",
    "season_Autumn",
    "season_Winter",
]

ARTIST_AUDIO_FEATURES = AUDIO_FEATURES + ARTIST_FEATURES

RANK_COLUMN = "Rank"
ID_COLUMN = "id"
DATE_COLUMN = "Date"
ARTIST_IDENTIFIER_COLUMN = "Artist (Ind.)"
NATIONALITY_COLUMN = "Nationality"
POINTS_TOTAL_COLUMN = "Points (Total)"
POINTS_INDIVIDUAL_COLUMN = "Points (Ind for each Artist/Nat)"
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_TO_INDEX = {season: idx for idx, season in enumerate(SEASON_ORDER)}


def load_classification_dataframe(
    data_path: Path,
    feature_cols: Iterable[str],
    rank_col: str,
    top_k: int,
    FEATURE_SET_NAME: str,
) -> tuple[pd.DataFrame, str]:
    logging.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    required_cols = list(feature_cols) + [rank_col, ID_COLUMN]

    # print(f"Initial dataset has {len(df)} songs.")

    # df = cleanData(df)

    # print(f"Dataset after cleaning has {len(df)} songs.")
    

    if FEATURE_SET_NAME == "Audio Features":
        print("Engineering Audio Features...")
        model_df = engineerAudioFeatures(df, required_cols)
    elif FEATURE_SET_NAME == "Audio and Artist Features":
        print("Engineering Audio and Artist Features...")
        model_df = engineerAudioAndArtistFeatures(
            df, feature_cols=feature_cols, rank_col=rank_col
        )
    elif FEATURE_SET_NAME == "Temporal Features":
        print("Engineering Temporal Features...")
        model_df = engineerTemporalFeatures(df, required_cols)
    else:
        raise ValueError(f"Unknown FEATURE_SET_NAME: {FEATURE_SET_NAME}")

    # model_df = (
    #     df[required_cols]
    #     .dropna(subset=required_cols)
    #     # .drop_duplicates()
    #     .copy()
    # )
    # # Normalize features to have mean 0 and std 1
    # model_df = normalize_features(model_df, feature_cols)

    target_col = f"is_top_{top_k}"
    model_df[target_col] = (model_df[rank_col] <= top_k).astype(int)
    model_df.reset_index(drop=True, inplace=True)
    logging.info(
        "Prepared dataframe with %d songs (positive rate %.2f%%).",
        len(model_df),
        100 * model_df[target_col].mean(),
    )
    return model_df, target_col

def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.rename(columns={'Artist (Ind.)': 'Artist_Ind', 'Points (Ind for each Artist/Nat)': 'Points_Ind', 'Points (Total)': 'Points_Tot'})

    df.drop_duplicates(inplace=True)

    # Find an example of a song that was removed from the dataset after cleaning
    # return all songs that were dropped from the dataset after cleaning
    


    cols_to_clean = ['# of Artist', '# of Nationality']
    for col in cols_to_clean:
    # 1. Remove text prefix and spaces (e.g., 'Artist 1' -> '1')
        df[col] = df[col].astype(str).str.replace('Artist ', '', regex=False)
        df[col] = df[col].astype(str).str.replace('Nationality ', '', regex=False)
    
    # 2. Convert to numeric integer
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    grouped_df = df.copy().reset_index(drop=True)

    # Group by song/date/id/url
    grouped_df = (
        df.groupby(["Title","Date", "id", "Song URL"], as_index=False)
        .agg({
            "Artists": lambda x: ", ".join(sorted(set(x))),  # merge unique artist names
            "Rank": "min",                                 # average rank if repeated
            "Danceability": "first",
            "Energy": "first",
            "Loudness": "first",
            "Speechiness": "first",
            "Acousticness": "first",
            "Instrumentalness": "first",
            "Valence": "first",
            "# of Artist": "max",
            "# of Nationality": "max",
            "Artist_Ind": lambda x: ", ".join(sorted(set(x))),  # merge individual artists
            "Nationality": lambda x: ", ".join(sorted(set(x))),    # merge nationalities
            "Continent": lambda x: ", ".join(sorted(set(x))),      # merge if varies
            "Points_Tot": "max",
            # "Points_Ind": "first" // NOT NEEDED!!
        })
        .copy()
        .reset_index(drop=True)
    )
    grouped_df['year'] = grouped_df['Date'].dt.year
    grouped_df['month'] = grouped_df['Date'].dt.month
    grouped_df['week'] = grouped_df['Date'].dt.isocalendar().week

    return grouped_df

def engineerAudioFeatures(
        df: pd.DataFrame,
        required_cols: Iterable[str],
    ) -> pd.DataFrame:
    # For audio features, all I need to do is exptract only requires columns, normalize then and return
    model_df = (
        df[required_cols]
        .dropna(subset=required_cols)
        .copy()
    )
    feature_cols = AUDIO_FEATURES

    # Normalize features to have mean 0 and std 1
    # model_df = normalize_features(model_df, feature_cols)
    return model_df

def engineerAudioAndArtistFeatures(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    rank_col: str,
) -> pd.DataFrame:

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df = df.rename(columns={'Artist (Ind.)': 'Artist_Ind', 'Points (Ind for each Artist/Nat)': 'Points_Ind', 'Points (Total)': 'Points_Tot'})

    # Drop duplicate rows (exact duplicates)
    df.drop_duplicates(inplace=True)

    cols_to_clean = ['# of Artist', '# of Nationality']
    for col in cols_to_clean:
        df[col] = df[col].astype(str).str.replace('Artist ', '', regex=False)
        df[col] = df[col].astype(str).str.replace('Nationality ', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    #Number
    num_artists_per_song = df.groupby('id')['Artist_Ind'].nunique().rename('Num_of_Artists_Total')
    df = df.merge(num_artists_per_song, on='id', how='left')

    # Cyclical encodings for temporal signals (mirrors artist notebook)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    check_count = df[df['Artist_Ind'].isin(['Frank Sinatra', 'Idina Menzel', 'Ozuna'])]
    check_count = check_count[['Artist_Ind', 'Nationality', 'Continent']]

    grouped = check_count.groupby(['Artist_Ind', 'Nationality','Continent']).size()

    top50_threshold = 180

    # Count number of days each song is in top 50
    song_days = df[df['Points_Tot'] > top50_threshold].groupby('id')['Date'].nunique()

    popular_songs = song_days[song_days >= 1].index
    df['Popular'] = df['id'].isin(popular_songs).astype(int)
    
    return df.copy()


def prepare_audio_artist_design_matrices(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    artist_col: str = "Artist_Ind",
    num_features: Iterable[str] | None = None,
    cat_features: Iterable[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build design matrices for the artist+audio configuration, mirroring the
    preprocessing in the artist notebook. Applies target encoding for artists
    using the training split and one-hot encodes nationality/continent.
    """

    if num_features is None:
        num_features = [
            "Danceability",
            "Energy",
            "Loudness",
            "Speechiness",
            "Acousticness",
            "Instrumentalness",
            "Valence",
            "Num_of_Artists_Total",
            "Month_sin",
            "Month_cos",
            "Day_sin",
            "Day_cos",
        ]
    if cat_features is None:
        cat_features = ['Nationality', 'Continent']

    artist_target_mean = train_df.groupby(artist_col)[target_col].mean()
    global_mean = float(train_df[target_col].mean())

    def _target_encode(df: pd.DataFrame) -> pd.Series:
        return df[artist_col].map(artist_target_mean).fillna(global_mean)

    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()
    train_df["Artist_TE"] = _target_encode(train_df)
    dev_df["Artist_TE"] = _target_encode(dev_df)
    test_df["Artist_TE"] = _target_encode(test_df)

    feature_cols = list(num_features) + ["Artist_TE"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    print(f"Training before fit: {train_df[cat_features].shape}")
    encoder.fit(train_df[cat_features])
    print(f"Training after fit: {train_df[cat_features].shape}")

    def _build_matrix(df: pd.DataFrame, isTrain=False) -> np.ndarray:
        X_num = df[feature_cols].to_numpy(dtype=float)
        print(f"Numeric feature matrix shape: ")
        print(X_num.shape)
        X_cat = encoder.transform(df[cat_features])
        print(f"Categorical feature matrix shape: ")
        print(X_cat.shape)
        return np.hstack([X_num, X_cat])

    matrices = {
        "train": (_build_matrix(train_df, isTrain=True), train_df[target_col].to_numpy()),
        "dev": (_build_matrix(dev_df), dev_df[target_col].to_numpy()),
        "test": (_build_matrix(test_df), test_df[target_col].to_numpy()),
    }
    matrices["categories_"] = encoder.categories_
    matrices["numeric_features"] = feature_cols
    matrices["categorical_features"] = cat_features
    return matrices

def engineerTemporalFeatures(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:

    engineered_df = df.copy()
    if DATE_COLUMN in engineered_df.columns:
        date_series = pd.to_datetime(
            engineered_df[DATE_COLUMN], dayfirst=True, errors="coerce"
        )
        engineered_df["year"] = date_series.dt.year
        engineered_df["month"] = date_series.dt.month
        engineered_df["month_sin"] = np.sin(2 * np.pi * engineered_df["month"] / 12)
        engineered_df["month_cos"] = np.cos(2 * np.pi * engineered_df["month"] / 12)
        engineered_df["season"] = pd.Categorical(
            engineered_df["month"].apply(_month_to_season),
            categories=SEASON_ORDER,
            ordered=True,
        )

        season_dummies = pd.get_dummies(
            engineered_df["season"], 
            prefix="season",
            dtype=float
        )
        for season in SEASON_ORDER:
            col_name = f"season_{season}"
            if col_name not in season_dummies.columns:
                season_dummies[col_name] = 0.0
        engineered_df = pd.concat([engineered_df, season_dummies], axis=1)

    engineered_df = _add_artist_metadata_features(engineered_df)

    if (
        RANK_COLUMN in engineered_df.columns
        and ID_COLUMN in engineered_df.columns
    ):
        numeric_ranks = pd.to_numeric(engineered_df[RANK_COLUMN], errors="coerce")
        engineered_df[RANK_COLUMN] = numeric_ranks
        top10_flag_col = "_top10_flag"
        top200_flag_col = "_top200_flag"
        engineered_df[top10_flag_col] = (numeric_ranks <= 10).astype(int)
        engineered_df[top200_flag_col] = (numeric_ranks <= 200).astype(int)

        # engineered_df["top10_day_count"] = (
        #     engineered_df.groupby(ID_COLUMN)[top10_flag_col]
        #     .transform("sum")
        #     .astype(float)
        # )
        # engineered_df["top200_day_count"] = (
        #     engineered_df.groupby(ID_COLUMN)[top200_flag_col]
        #     .transform("sum")
        #     .astype(float)
        # )

        if "season" in engineered_df.columns:
            top10_season_map = (
                engineered_df[engineered_df[top10_flag_col] == 1]
                .groupby(ID_COLUMN)["season"]
                .agg(_season_mode_or_na)
            )
            engineered_df["top10_season"] = engineered_df[ID_COLUMN].map(
                top10_season_map
            )
        else:
            engineered_df["top10_season"] = pd.NA

        engineered_df["top10_season_index"] = (
            engineered_df["top10_season"]
            .map(SEASON_TO_INDEX)
            .astype(float)
            .fillna(-1)
        )

        engineered_df.drop(columns=[top10_flag_col, top200_flag_col], inplace=True)

    engineered_df = engineered_df[required_cols].dropna(subset=required_cols).copy()
    return engineered_df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    normalized_df = df.copy()
    for col in feature_cols:
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        if std_val and not np.isclose(std_val, 0):
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = normalized_df[col] - mean_val
    return normalized_df


def create_classification_splits(
    df: pd.DataFrame,
    target_col: str,
    holdout_fraction: float,
    dev_share: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info(
        "Creating splits with holdout_fraction=%.2f and dev_share=%.2f",
        holdout_fraction,
        dev_share,
    )
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Expected {ID_COLUMN!r} column to partition by song id.")
    if df[ID_COLUMN].isna().any():
        raise ValueError("Song id column contains missing values; cannot build disjoint splits.")

    id_target = (
        df.groupby(ID_COLUMN)[target_col]
        .max()
        .reset_index()
        .rename(columns={target_col: "song_target"})
    )

    id_train, id_holdout = train_test_split(
        id_target,
        test_size=holdout_fraction,
        stratify=id_target["song_target"],
        random_state=random_state,
    )
    id_dev, id_test = train_test_split(
        id_holdout,
        train_size=dev_share,
        stratify=id_holdout["song_target"],
        random_state=random_state,
    )

    train_df = df[df[ID_COLUMN].isin(id_train[ID_COLUMN])]
    dev_df = df[df[ID_COLUMN].isin(id_dev[ID_COLUMN])]
    test_df = df[df[ID_COLUMN].isin(id_test[ID_COLUMN])]

    _ensure_disjoint_song_ids(train_df, dev_df, test_df)
    _assert_class_presence(train_df, target_col, split_name="train")
    _assert_class_presence(dev_df, target_col, split_name="dev")
    _assert_class_presence(test_df, target_col, split_name="test")
    # _ensure_similar_positive_rates(
    #     {"train": train_df, "dev": dev_df, "test": test_df},
    #     target_col=target_col,
    #     tolerance=0.001,
    # )

    logging.info(
        "Split sizes -> train=%d, dev=%d, test=%d",
        len(train_df),
        len(dev_df),
        len(test_df),
    )
    for split_name, split_df in [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]:
        positive_rate = split_df[target_col].mean()
        logging.info(
            "%s split positive rate: %.2f%%",
            split_name.capitalize(),
            100 * positive_rate,
        )
    return train_df, dev_df, test_df


def create_classification_splits_cross(
    df: pd.DataFrame,
    target_col: str,
    *,
    holdout_fraction: float = 0.25,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into cross-validation (train+cv) and holdout partitions.

    The split is stratified by song id to prevent leakage between partitions.
    """
    logging.info(
        "Creating cross/holdout splits with holdout_fraction=%.2f",
        holdout_fraction,
    )
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Expected {ID_COLUMN!r} column to partition by song id.")
    if df[ID_COLUMN].isna().any():
        raise ValueError("Song id column contains missing values; cannot build disjoint splits.")

    id_target = (
        df.groupby(ID_COLUMN)[target_col]
        .max()
        .reset_index()
        .rename(columns={target_col: "song_target"})
    )

    id_cross, id_holdout = train_test_split(
        id_target,
        test_size=holdout_fraction,
        stratify=id_target["song_target"],
        random_state=random_state,
    )

    cross_df = df[df[ID_COLUMN].isin(id_cross[ID_COLUMN])]
    holdout_df = df[df[ID_COLUMN].isin(id_holdout[ID_COLUMN])]

    if set(cross_df[ID_COLUMN]) & set(holdout_df[ID_COLUMN]):
        raise ValueError("Cross-validation and holdout splits share song ids.")

    _assert_class_presence(cross_df, target_col, split_name="cross")
    _assert_class_presence(holdout_df, target_col, split_name="holdout")

    logging.info(
        "Split sizes -> cross=%d, holdout=%d",
        len(cross_df),
        len(holdout_df),
    )
    for split_name, split_df in [
        ("cross", cross_df),
        ("holdout", holdout_df),
    ]:
        logging.info(
            "%s split positive rate: %.2f%%",
            split_name.capitalize(),
            100 * split_df[target_col].mean(),
        )

    return cross_df, holdout_df


def _ensure_disjoint_song_ids(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    def _id_set(split: pd.DataFrame) -> set:
        return set(split[ID_COLUMN].dropna())

    train_ids = _id_set(train_df)
    dev_ids = _id_set(dev_df)
    test_ids = _id_set(test_df)

    overlaps = {
        ("train", "dev"): train_ids & dev_ids,
        ("train", "test"): train_ids & test_ids,
        ("dev", "test"): dev_ids & test_ids,
    }
    offenders = {pair: ids for pair, ids in overlaps.items() if ids}
    if offenders:
        details = "; ".join(
            f"{a}-{b}: {sorted(list(ids))[:5]}"
            for (a, b), ids in offenders.items()
        )


def _assert_class_presence(
    split: pd.DataFrame, target_col: str, *, split_name: str
) -> None:
    classes_present = split[target_col].dropna().unique()
    if len(classes_present) < 2:
        raise ValueError(
            f"Split '{split_name}' does not contain both positive and negative examples."
        )


def _ensure_similar_positive_rates(
    splits: dict[str, pd.DataFrame],
    *,
    target_col: str,
    tolerance: float = 0.001,
) -> None:
    """
    Ensure all splits have nearly identical positive-class ratios.

    Parameters
    ----------
    splits:
        Mapping from split name to dataframe.
    target_col:
        Column holding the binary target.
    tolerance:
        Maximum allowed absolute difference between any two positive ratios.
    """

    rates = {
        name: split[target_col].mean()
        for name, split in splits.items()
    }
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    print(f"Positive class rates across splits: {rates}")
    print(f"Max rate: {max_rate}, Min rate: {min_rate}, Difference: {max_rate - min_rate}")
    if (max_rate - min_rate) > tolerance:
        detail = ", ".join(f"{name}={rate:.3f}" for name, rate in rates.items())
        raise ValueError(
            "Positive-class fractions differ across splits beyond tolerance "
            f"{tolerance:.3f}: {detail}"
        )
        raise ValueError(
            "Song ids overlap across splits, violating disjointness: " + details
        )


def _month_to_season(month_value):
    if pd.isna(month_value):
        return pd.NA
    month = int(month_value)
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Winter"


def _season_mode_or_na(series: pd.Series):
    if series.empty:
        return pd.NA
    mode = series.mode()
    if mode.empty:
        return pd.NA
    return mode.iloc[0]


def _add_artist_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered_df = df.copy()

    def _find_first_existing(columns: Iterable[str]) -> str | None:
        for col in columns:
            if col in engineered_df.columns:
                return col
        return None

    if "artist_count" not in engineered_df.columns:
        source = _find_first_existing(["# of Artist", "artist_count"])
        if source:
            engineered_df["artist_count"] = pd.to_numeric(
                engineered_df[source], errors="coerce"
            )
        else:
            artist_col = _find_first_existing(["Artist_Ind", ARTIST_IDENTIFIER_COLUMN])
            if artist_col:
                engineered_df["artist_count"] = (
                    engineered_df[artist_col]
                    .fillna("")
                    .apply(
                        lambda val: len(
                            {
                                name.strip()
                                for name in str(val).split(",")
                                if name.strip()
                            }
                        )
                    )
                    .astype(float)
                )
            else:
                engineered_df["artist_count"] = np.nan

    if "nationality_count" not in engineered_df.columns:
        source = _find_first_existing(["# of Nationality", "nationality_count"])
        if source:
            engineered_df["nationality_count"] = pd.to_numeric(
                engineered_df[source], errors="coerce"
            )
        else:
            nationality_col = _find_first_existing(["Nationality", NATIONALITY_COLUMN])
            if nationality_col:
                engineered_df["nationality_count"] = (
                    engineered_df[nationality_col]
                    .fillna("")
                    .apply(
                        lambda val: len(
                            {
                                name.strip()
                                for name in str(val).split(",")
                                if name.strip()
                            }
                        )
                    )
                    .astype(float)
                )
            else:
                engineered_df["nationality_count"] = np.nan

    # if "points_total" not in engineered_df.columns:
    #     source = _find_first_existing(
    #         ["Points_Tot", POINTS_TOTAL_COLUMN, "Points (Total)"]
    #     )
    #     if source:
    #         engineered_df["points_total"] = pd.to_numeric(
    #             engineered_df[source], errors="coerce"
    #         )
    #     else:
    #         engineered_df["points_total"] = np.nan

    # if "points_individual" not in engineered_df.columns:
    #     source = _find_first_existing(
    #         ["Points_Ind", POINTS_INDIVIDUAL_COLUMN, "Points (Ind for each Artist/Nat)"]
    #     )
    #     if source:
    #         engineered_df["points_individual"] = pd.to_numeric(
    #             engineered_df[source], errors="coerce"
    #         )
    #     else:
    #         engineered_df["points_individual"] = np.nan

    return engineered_df
