import base64
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from numpy import interp
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))
# appId

df = pd.read_csv("sentiment analysis result.csv")
app_id = [
    ("ANZ Australia (com.anz.android.gomoney)", "com.anz.android.gomoney"),
    ("ANZ Direct Auth (com.anz.anzpacific)", "com.anz.anzpacific"),
    (
        "ANZ Transactive - Global (com.anz.transactive.global)",
        "com.anz.transactive.global",
    ),
    ("Z Digital Key (com.anz.DigitalKey)", "com.anz.DigitalKey"),
]


def mapping(val):
    res = interp(val, [-1, 1], [0, 5])
    g = float("{0:.3f}".format(res))
    return g


def first_bar(app):
    senti = df["Sentiment_Analysis"]
    apps = df["appId"]
    positive = 0
    negative = 0
    neutral = 0
    for i in range(len(senti)):
        if apps[i] == app:
            if senti[i] == "Positive":
                positive = positive + 1
            if senti[i] == "Negative":
                negative = negative + 1
            if senti[i] == "Neutral":
                neutral = neutral + 1
    res = [positive, neutral, negative]

    return res


def second_graph(app):
    score = df["Sentiment_Analysis_value"]
    apps = df["appId"]
    senti = df["Sentiment_Analysis"]
    very_pos = 0
    pos = 0
    very_neg = 0
    neg = 0
    neu = 0
    for i in range(len(score)):
        if apps[i] == app:
            if senti[i] == "Positive":
                if score[i] >= 0.5:
                    very_pos = very_pos + 1
                else:
                    pos = pos + 1
            if senti[i] == "Negative":
                if score[i] < (-0.5):
                    very_neg = very_neg + 1
                else:
                    neg = neg + 1
            if senti[i] == "Neutral":
                neu = neu + 1
    res = [very_pos, pos, neu, neg, very_neg]

    return res


def third_graph(app):
    print("app id :", app)
    con = df["content"]
    apps = df["appId"]
    print("lenght of apps :", len(apps))
    print("lenght of con :", len(con))
    content = []
    for i in range(len(con)):
        if apps[i] == app:
            a = con[i]
            content.append(a)
    print("lenght of content :", len(content))
    filter_word = []
    for word in content:
        words = word_tokenize(word)
        for i in words:
            if i.casefold() not in stop_words:
                filter_word.append(i)
    print("length of filter word :", len(filter_word))
    pure_word = []
    bad_words = ["...", ".", "..", ",", "''", "'", "?", "$", ""]
    for item in filter_word:
        if item not in bad_words and not item.isnumeric():
            if len(item) > 2:
                pure_word.append(item)
    positive_words = []
    negative_words = []
    neutral_words = []
    for i in pure_word:
        a = sia.polarity_scores(i)
        if a["compound"] == 0:
            neutral_words.append(i)
        if a["compound"] > 0:
            positive_words.append(i)
        if a["compound"] < 0:
            negative_words.append(i)
    print("lenght of pure words :", len(pure_word))
    frequency_distribution = FreqDist(pure_word)
    frequency = frequency_distribution.most_common(80)
    print(frequency)

    res = [positive_words, negative_words, neutral_words, frequency]

    return res


def most_frequent_word(frequency):
    name = []
    count = []
    for i in range(20):
        a = frequency[i]
        name.append(a[0])
        count.append(a[1])
    return name, count


def tacker_of_app(app):
    print(app)
    positive = 0
    negative = 0
    neutral = 0
    senti = df["Sentiment_Analysis"]
    apps = df["appId"]
    for i in range(len(df["appId"])):
        if apps[i] == app:
            if senti[i] == "Positive":
                positive = positive + 1
            if senti[i] == "Negative":
                negative = negative + 1
            if senti[i] == "Neutral":
                neutral = neutral + 1
    if positive > 0 or negative > 0 or neutral > 0:
        res = float((positive - negative) / (positive + negative + neutral))
    else:
        res = 0
    g = float("{0:.3f}".format(res))
    g = mapping(g)

    return g


def niddle_position(a):
    if a > 0 and a < 1:
        p = a * 10
        x = float((0.62 - 0.5) / 10)
        y = float(0.5 + (x * p))
        y = float("{0:.3f}".format(y))
        f = 0.2
        return a, f, y
    if a >= 1 and a < 2:
        p = float((a - 1) * 10)
        print("points :", p)
        x = float((0.225 - 0.175) / 10)
        x = float("{0:.3f}".format(x))
        print("value of x :", x)
        y = float(0.175 + (x * p))
        y = float("{0:.3f}".format(y))
        l = 0.7

        return a, y, l
    if a >= 2 and a < 3:
        p = float((a - 2) * 10)
        x = float((0.262 - 0.222) / 10)
        x = float("{0:.3f}".format(x))
        y = float(0.222 + (x * p))
        y = float("{0:.3f}".format(y))
        l = 0.78
        return a, y, l

    if a >= 3 and a < 4:
        p = float((a - 3) * 10)
        x = float((0.85 - 0.6) / 10)
        x = float("{0:.3f}".format(x))
        y = float(0.85 - (x * p))
        y = float("{0:.3f}".format(y))
        f = 0.269
        return a, f, y
    if a >= 4 and a <= 5:
        p = float((a - 4) * 10)
        x = float((0.69 - 0.5) / 10)
        x = float("{0:.3f}".format(x))
        y = float(0.69 - (x * p))
        y = float("{0:.3f}".format(y))
        f = 0.3
        return a, f, y


def generate_wordcloud(app, pos, neg):

    # print(words)
    # word = words['frequency']
    # print(len(word))
    print("lenght of negative words :", len(neg))

    name = []
    name2 = []
    count = []
    if len(pos) < 80:
        for i in range(len(pos)):
            a = pos[i]
            name.append(a)
    else:
        for i in range(80):
            a = pos[i]
            name.append(a)
    if len(neg) < 80:
        for i in range(len(neg)):
            b = neg[i]
            name2.append(b)
    else:
        for i in range(80):
            b = neg[i]
            name2.append(b)
    unique_string = (" ").join(name)
    unique_string2 = (" ").join(name2)
    print(unique_string)
    print(unique_string2)
    print(os.getcwd())
    base = os.path.join(os.getcwd(), "assets")
    print(base)
    dir_list = os.listdir(base)
    print(dir_list)
    # if len(dir_list) > 0:
    #     p = os.path.join(base, dir_list[0])
    #     q=os.path.join(base,dir_list[1])
    #
    #     os.remove(p)
    #     os.remove(q)
    p = app + "-positive.png"
    q = app + "-negative.png"
    if p in dir_list:
        pass
    else:
        print("generating the word cloud ")
        wordcloud = WordCloud(
            max_font_size=50,
            max_words=100,
            background_color="white",
            width=1000,
            height=500,
        ).generate(unique_string)
        wordcloud.to_file(f"assets/{app}-positive.png")
    if q in dir_list:
        pass
    else:
        wordcloud2 = WordCloud(
            max_font_size=50,
            max_words=100,
            background_color="white",
            width=1000,
            height=500,
        ).generate(unique_string2)
        wordcloud2.to_file(f"assets/{app}-negative.png")

    return "okay"


app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children="Select App to see the user sentiment analysis "),
    html.Div([
        dcc.Dropdown(
            id="apps_ids",
            options=[{
                "label": x,
                "value": y
            } for x, y in app_id],
            value=None,
        )
    ]),
    dcc.Graph(id="graph1"),
    dcc.Graph(id="graph2"),
    dcc.Graph(id="graph3"),
    dcc.Graph(id="graph4"),
    html.H1(children="Tracker and Performance of the app "),
    dcc.Graph(id="graph6"),
    dcc.Graph(id="graph5"),
    html.H1(children="word cloud"),
    html.Div(
        children="""1 .Positive  Word Cloud""",
        style={
            "color": "green",
            "fontSize": 14
        },
    ),
    html.Div(id="positive"),
    html.P(
        children="""2 .Negative  Word Cloud""",
        style={
            "color": "red",
            "fontSize": 14
        },
    ),
    html.Div(id="negative"),
])


@app.callback(
    Output("graph1", "figure"),
    Output("graph2", "figure"),
    Output("graph3", "figure"),
    Output("graph4", "figure"),
    Output("graph5", "figure"),
    Output("graph6", "figure"),
    Output(component_id="positive", component_property="children"),
    Output(component_id="negative", component_property="children"),
    Input("apps_ids", "value"),
)
def make_graphs(apps):
    print(apps)

    # this is for the first graph
    first = first_bar(apps)
    fig = go.Figure(
        go.Bar(
            x=["Positive", "Neutral", "Negative"],
            y=first,
            width=[0.2, 0.2, 0.2],
            marker_color=["green", "gold", "red"],
        ))
    fig.update_layout(
        title_text="1 . Sentiment analysis on the basis of Scores ")
    fig.update_layout(barmode="group")

    # this is for the second graph
    second = second_graph(apps)
    fig2 = go.Figure(
        go.Bar(
            x=[
                "Very Postive", "Postive", "Neutral", "Negative",
                "Very Negative"
            ],
            y=second,
            width=[0.2, 0.2, 0.2, 0.2, 0.2],
            marker_color=["darkgreen", "green", "orange", "pink", "red"],
        ))
    fig2.update_layout(
        title_text="2 . Sentiment analysis on the basis of thumbsUpCount  ")
    fig2.update_layout(barmode="group")

    # this is for the pie chard on the basis of words
    third = third_graph(apps)
    daud = generate_wordcloud(apps, third[0], third[1])
    print("lenght of third :", len(third))
    print(type(third[3]))
    print("lenght of mfw :", len(third[3]))
    postive_word = len(third[0])
    negative_word = len(third[1])
    neutral_word = len(third[2])
    colors = ["green", "red", "gold"]
    fig3 = go.Figure(
        go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[postive_word, negative_word, neutral_word],
        ))
    fig3.update_traces(
        hoverinfo="label+percent",
        textinfo="value",
        textfont_size=20,
        marker=dict(colors=colors, line=dict(color="#000000", width=2)),
    )
    fig3.update_layout(
        title_text="1 . Sentiment analysis % on the basis of ContentReplies  ")

    # this section is for the pie chart of on the basis of reviews

    fig4 = go.Figure(
        go.Pie(labels=["Positive", "Negative", "Neutral"], values=first))
    fig4.update_traces(
        hoverinfo="label+percent",
        textinfo="value",
        textfont_size=20,
        marker=dict(colors=colors, line=dict(color="#000000", width=2)),
    )
    fig4.update_layout(
        title_text=
        "2 . Sentiment analysis % break down on the basis of Reviews  ")

    # this section is for the most frequent word used in the content of the app

    name, count = most_frequent_word(third[3])
    fig5 = go.Figure(go.Bar(x=name, y=count))
    fig5.update_layout(yaxis=dict(
        title="No of times of occurance",
        titlefont_size=16,
        tickfont_size=14,
    ))
    fig5.update_layout(title_text="most frequent words ")

    # this is for the tracker
    tra = tacker_of_app(apps)
    val, f, l = niddle_position(tra)
    base_chart = {
        "values": [40, 10, 10, 10, 10, 10, 10],
        "labels": ["-", "0", "1", "2", "3", "4", "5"],
        "domain": {
            "x": [0, 0.48]
        },
        "marker": {
            "colors": [
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
                "rgb(255, 255, 255)",
            ],
            "line": {
                "width": 50
            },
        },
        "name": "Gauge",
        "hole": 0.4,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 108,
        "showlegend": False,
        "hoverinfo": "none",
        "textinfo": "label",
        "textposition": "outside",
    }

    meter_chart = {
        "values": [50, 10, 10, 10, 10, 10],
        "labels": [
            "performance",
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
        "marker": {
            "colors": [
                "rgb(255, 255, 255)",
                "rgb(246,30,30)",
                "rgb(255,128,0)",
                "rgb(245,215,26)",
                "rgb(178,255,102)",
                "rgb(0,204,0)",
            ]
        },
        "domain": {
            "x": [0, 0.48]
        },
        "name":
        "Gauge",
        "hole":
        0.3,
        "type":
        "pie",
        "direction":
        "clockwise",
        "rotation":
        90,
        "showlegend":
        False,
        "textinfo":
        "label",
        "textposition":
        "inside",
        "hoverinfo":
        "none",
    }

    layout = {
        "xaxis": {
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis": {
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        "shapes": [{
            "type": "path",
            "path": f"M 0.235 0.5 L {f}  {l} L 0.245 0.5 Z",
            "fillcolor": "rgba(44, 160, 101, 0.5)",
            "line": {
                "width": 1
            },
            "xref": "paper",
            "yref": "paper",
        }],
        "annotations": [{
            "xref": "paper",
            "yref": "paper",
            "x": 0.23,
            "y": 0.45,
            "text": val,
            "showarrow": False,
        }],
    }

    # we don't want the boundary now
    base_chart["marker"]["line"]["width"] = 0

    fig6 = {"data": [base_chart, meter_chart], "layout": layout}

    path = os.path.join(os.getcwd(), "assets")
    img1 = apps + "-positive.png"
    img2 = apps + "-negative.png"
    image_filename = os.path.join(path, img1)  # replace with your own image
    image_filename2 = os.path.join(path, img2)

    def b64_image(image_filename):
        with open(image_filename, "rb") as f:
            image = f.read()
        return "data:image/png;base64," + base64.b64encode(image).decode(
            "utf-8")

    image1 = (html.Img(src=b64_image(image_filename),
                       height="250",
                       width="700"), )
    image2 = html.Img(src=b64_image(image_filename2),
                      height="250",
                      width="700")

    return fig, fig2, fig3, fig4, fig5, fig6, image1, image2


if __name__ == "__main__":
    app.run_server(debug=True)
