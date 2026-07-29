import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI
from loguru import logger

from database import postgres_connection
from schema import PostGet


# =========================
# LOAD DATA
# =========================

def load_sql(query: str) -> pd.DataFrame:
    conn = postgres_connection()
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    return df


def load_model(path="model_regressor.pkl"):

    if os.environ.get("IS_LMS", "0") == "1":
        path = os.environ["MODEL_PATH"]

    with open(path, "rb") as f:
        model = pickle.load(f)

    logger.success("Model loaded")
    return model


# =========================
# INIT
# =========================

app = FastAPI()

model = load_model()

user_features = load_sql(
    """
    SELECT * FROM ariana13_user_features
    """
)

post_features = load_sql(
    """
    SELECT * FROM ariana13_post_features
    """
)

logger.success("Data loaded")


# =========================
# FEATURES
# =========================

FEATURES = [
    "hour",
    "month",
    "day",
    "os",
    "source",
    "age_group",
    "country_city",
    "text",
    "topic",
    "text_len",
    "user_topic_views"
]


CAT_FEATURES = [  
    "country_city",
    "topic","os","age_group","source"
]


TEXT_FEATURES = [
    "text"
]


def get_part_of_day(hour):

    if 6 <= hour < 12:
        return "morning"

    if 12 <= hour < 18:
        return "afternoon"

    if 18 <= hour < 24:
        return "evening"

    return "night"



# =========================
# ENDPOINT
# =========================

@app.get(
    "/post/recommendations/",
    response_model=List[PostGet]
)
def recommended_posts(
        user_id: int,
        dt: datetime,
        limit: int = 10
):

    logger.info(
        f"user_id={user_id}, dt={dt}"
    )


    # ---------------------
    # POSTS
    # ---------------------

    DataTemp = post_features.copy()


    # ---------------------
    # USER FEATURES
    # ---------------------

    user_row = user_features[
        user_features["user_id"] == user_id
    ].iloc[0]


    user_cols = [
        "os",
        "source",
        "age_group",
        "country_city"
    ]


    for col in user_cols:
        DataTemp[col] = user_row[col]


    # ---------------------
    # TIME FEATURES
    # ---------------------

    DataTemp["hour"] = dt.hour
    DataTemp["month"] = dt.month
    DataTemp["day"] = dt.day

    #DataTemp["part_of_day"] = get_part_of_day(
    #    dt.hour
    #)


    # ---------------------
    # TYPES
    # ---------------------

    for col in CAT_FEATURES:
        DataTemp[col] = (
            DataTemp[col]
            .fillna("unknown")
            .astype(str)
        )


    DataTemp["text"] = (
        DataTemp["text"]
        .fillna("")
        .astype(str)
    )


    # сохраняем id постов для ответа
    post_ids = DataTemp["post_id"].copy()


    # оставляем только признаки для модели
    X_predict = DataTemp[FEATURES]


    prediction = model.predict(X_predict)


    # возвращаем post_id обратно
    DataTemp["prediction"] = prediction
    DataTemp["post_id"] = post_ids


    result = (
        DataTemp
        .sort_values(
            "prediction",
            ascending=False
        )
        .head(limit)
    )


    recs = [
        PostGet(
            id=row["post_id"],
            text=row["text"],
            topic=row["topic"]
        )
        for _, row in result.iterrows()
    ]


    return recs