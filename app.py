import os

import joblib
import pandas as pd
import streamlit as st

from scipy.sparse import load_npz

from supabase import create_client


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = st.secrets[
    "SUPABASE_URL"
]

SUPABASE_KEY = st.secrets[
    "SUPABASE_KEY"
]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)



ARTIFACT_DIR = "artifacts"


@st.cache_data
def load_books():

    books = pd.read_parquet(
        os.path.join(
            ARTIFACT_DIR,
            "books.parquet"
        )
    )

    books["genre_list"] = (
        books["genre_list"]
        .fillna("")
        .apply(
            lambda x: [
                g
                for g in str(x).split("||")
                if g
            ]
        )
    )

    return books


@st.cache_resource
def load_model():

    model = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "nearest_neighbors.joblib"
        )
    )

    content_matrix = load_npz(
        os.path.join(
            ARTIFACT_DIR,
            "content_matrix.npz"
        )
    )

    return model, content_matrix

def search_books(query, limit=30):
    if not query or not query.strip():
        return books.iloc[0:0].copy()

    query = query.strip().lower()

    title_match = (
        books["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(query, regex=False)
    )

    author_match = (
        books["author"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(query, regex=False)
    )

    results = books[
        title_match | author_match
    ].copy()

    return results.head(limit)
books = load_books()

model, content_matrix = (
    load_model()
)


# ============================================================
# SESSION STATE
# ============================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "source_book_id" not in st.session_state:
    st.session_state.source_book_id = None


# ============================================================
# AUTHENTICATION
# ============================================================

def login():

    st.title(
        "📚 Book Recommender"
    )

    login_tab, signup_tab = st.tabs(
        [
            "Login",
            "Create account"
        ]
    )


    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    with login_tab:

        email = st.text_input(
            "Email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            type="primary"
        ):

            try:

                response = (
                    supabase
                    .auth
                    .sign_in_with_password(
                        {
                            "email": email,
                            "password": password
                        }
                    )
                )

                st.session_state.user = (
                    response.user
                )

                st.success(
                    "Successfully logged in."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Login failed: {e}"
                )


    # --------------------------------------------------------
    # SIGN UP
    # --------------------------------------------------------

    with signup_tab:

        email = st.text_input(
            "Email",
            key="signup_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        if st.button(
            "Create account"
        ):

            try:

                response = (
                    supabase
                    .auth
                    .sign_up(
                        {
                            "email": email,
                            "password": password
                        }
                    )
                )

                st.success(
                    "Account created. "
                    "Check your email if confirmation "
                    "is enabled in Supabase."
                )

            except Exception as e:

                st.error(
                    f"Could not create account: {e}"
                )


if st.session_state.user is None:

    login()

    st.stop()


user = st.session_state.user

user_id = user.id


# ============================================================
# SEARCH
# ============================================================

query = st.text_input(
    "Search for a book",
    placeholder="Enter a book title, e.g. Inferno"
)

if query:

    results = search_books(query, limit=30)

    if results.empty:

        st.warning(
            "No books found."
        )

    else:

        st.write(
            f"Found {len(results)} matching books:"
        )

        for _, book in results.iterrows():

            book_id = int(book["book_id"])

            # ------------------------------------------------
            # EACH BOOK HAS ITS OWN CONTAINER
            # ------------------------------------------------

            with st.container(border=True):

                st.subheader(
                    book["title"]
                )

                st.write(
                    f"**Author:** {book['author']}"
                )

                if pd.notna(
                    book["rating_clean"]
                ):

                    st.write(
                        f"Goodreads rating: "
                        f"{book['rating_clean']:.2f} ⭐"
                    )

                if book["genre_list"]:

                    st.write(
                        "Genres: "
                        + ", ".join(
                            book["genre_list"]
                        )
                    )

                # ------------------------------------------------
                # THIS IS THE IMPORTANT PART
                # ------------------------------------------------

                if st.button(
                    "Select this book",
                    key=f"select_book_{book_id}"
                ):

                    st.session_state[
                        "source_book_id"
                    ] = book_id

                    st.session_state[
                        "selected_book"
                    ] = book

                    st.success(
                        f"Selected: "
                        f"{book['title']} — "
                        f"{book['author']}"
                    )

                    st.rerun()

# ============================================================
# RECOMMENDATIONS
# ============================================================

def recommend_books(
    book_id,
    n=10
):

    matches = books.index[
        books["book_id"] == book_id
    ]

    if len(matches) == 0:

        return books.iloc[0:0]


    row_position = (
        books.index.get_loc(
            matches[0]
        )
    )


    distances, indices = (
        model.kneighbors(
            content_matrix[
                row_position
            ],
            n_neighbors=min(
                n + 1,
                len(books)
            )
        )
    )


    recommendations = (
        books.iloc[
            indices[0]
        ].copy()
    )


    recommendations[
        "similarity"
    ] = 1 - distances[0]


    recommendations = (
        recommendations[
            recommendations["book_id"]
            != book_id
        ]
    )


    return recommendations.head(n)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📚 Book Recommender"
    )

    st.write(
        f"Logged in as:"
    )

    st.caption(
        user.email
    )


    page = st.radio(
        "Menu",
        [
            "🏠 Home",
            "🔎 Find a Book",
            "📖 My Books",
            "🎯 Recommendations",
            "⭐ My Ratings"
        ]
    )


    if st.button(
        "Logout"
    ):

        supabase.auth.sign_out()

        st.session_state.user = None

        st.session_state.source_book_id = None

        st.rerun()


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "📚 Goodreads Book Recommender"
    )

    st.write(
        """
        Search for a book and receive recommendations
        based on the trained recommendation model.
        """
    )

    st.info(
        """
        Books are identified using a unique `book_id`.
        The title alone is never used as the identity.

        Therefore books with the same title but different
        authors remain separate.
        """
    )


# ============================================================
# FIND BOOK
# ============================================================

elif page == "🔎 Find a Book":

    st.title(
        "🔎 Find a Book"
    )

    query = st.text_input(
        "Search by title or author",
        placeholder="Try: Inferno"
    )


    results = search_books(
        query
    )


    if query and results.empty:

        st.warning(
            "No books found."
        )


    if not results.empty:

        # IMPORTANT:
        # The value displayed to the user contains
        # BOTH title and author.

        book_options = {}

        book_options = {}

    for _, row in results.iterrows():
    
        title = str(row["title"])
        author = str(row["author"])
    
        book_format = row.get(
            "bookformat",
            ""
        )
    
        if pd.isna(book_format):
            book_format = ""
    
        book_format = str(
            book_format
        ).strip()
    
        if book_format:
    
            label = (
                f"{title} — "
                f"{author} — "
                f"{book_format}"
            )
    
        else:
    
            label = (
                f"{title} — "
                f"{author}"
            )
    
        book_options[
            int(row["book_id"])
        ] = label


        selected_id = st.selectbox(
            "Select the exact book",
            list(
                book_options.keys()
            ),
            format_func=lambda x:
                book_options[x]
        )


        selected = books.loc[
            books["book_id"] == selected_id
        ].iloc[0]


        st.divider()


        st.subheader(
            selected["title"]
        )

        st.write(
            f"**Author:** "
            f"{selected['author']}"
        )


        if pd.notna(
            selected["pages_clean"]
        ):

            st.write(
                f"**Pages:** "
                f"{int(selected['pages_clean'])}"
            )


        if pd.notna(
            selected["rating_clean"]
        ):

            st.write(
                f"**Goodreads rating:** "
                f"{selected['rating_clean']:.2f}"
            )


        if selected["genre_list"]:

            st.write(
                f"**Genres:** "
                f"{', '.join(selected['genre_list'])}"
            )


        if st.button(
            "Use this book",
            type="primary"
        ):

            st.session_state.source_book_id = (
                selected_id
            )

            st.success(
                "Book selected!"
            )

if "source_book_id" in st.session_state:

    selected_id = (
        st.session_state["source_book_id"]
    )

    selected_book = books[
        books["book_id"] == selected_id
    ].iloc[0]

    st.divider()

    st.subheader(
        "Selected book"
    )

    st.write(
        f"**{selected_book['title']}** "
        f"— {selected_book['author']}"
    )

    if st.button(
        "📚 Get recommendations",
        type="primary"
    ):

        recommendations = recommend_books(
            selected_id,
            n=10
        )

        st.session_state[
            "recommendations"
        ] = recommendations

        st.rerun()

# ============================================================
# MY BOOKS
# ============================================================

elif page == "📖 My Books":

    st.title(
        "📖 My Books"
    )

    st.write(
        "Add books you have read and rate them."
    )


    query = st.text_input(
        "Find a book",
        key="my_books_query"
    )


    results = search_books(
        query
    )


    if not results.empty:

        options = {}

        for _, row in results.iterrows():

            options[
                int(row["book_id"])
            ] = (
                f"{row['title']} "
                f"— {row['author']}"
            )


        selected_id = st.selectbox(
            "Book",
            list(options.keys()),
            format_func=lambda x:
                options[x]
        )


        rating = st.slider(
            "Your rating",
            min_value=1,
            max_value=5,
            value=5
        )


        if st.button(
            "Save book"
        ):

            try:

                # Reading history
                supabase.table(
                    "reading_history"
                ).upsert(
                    {
                        "user_id": user_id,
                        "book_id": int(
                            selected_id
                        ),
                        "status": "read"
                    },
                    on_conflict=(
                        "user_id,book_id"
                    )
                ).execute()


                # User rating
                supabase.table(
                    "book_ratings"
                ).upsert(
                    {
                        "user_id": user_id,
                        "book_id": int(
                            selected_id
                        ),
                        "rating": int(
                            rating
                        )
                    },
                    on_conflict=(
                        "user_id,book_id"
                    )
                ).execute()


                st.success(
                    "Book and rating saved!"
                )


            except Exception as e:

                st.error(
                    f"Could not save: {e}"
                )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":

    st.title(
        "🎯 Recommendations"
    )


    if (
        st.session_state.source_book_id
        is None
    ):

        st.warning(
            """
            First go to "Find a Book" and
            select a book.
            """
        )

        st.stop()


    source = books.loc[
        books["book_id"]
        == st.session_state.source_book_id
    ].iloc[0]


    st.write(
        f"Recommendations based on:"
    )

    st.subheader(
        f"{source['title']} — "
        f"{source['author']}"
    )


    recommendations = (
        recommend_books(
            st.session_state.source_book_id,
            n=10
        )
    )


    for _, row in recommendations.iterrows():

        book_id = int(
            row["book_id"]
        )


        with st.container(
            border=True
        ):

            st.subheader(
                row["title"]
            )

            st.write(
                f"**Author:** "
                f"{row['author']}"
            )

            st.write(
                f"**Similarity:** "
                f"{row['similarity']:.1%}"
            )


            if row["genre_list"]:

                st.write(
                    f"**Genres:** "
                    f"{', '.join(row['genre_list'])}"
                )


            rating = st.radio(
                "How would you rate this recommendation?",
                [
                    1,
                    2,
                    3,
                    4,
                    5
                ],
                horizontal=True,
                key=f"rating_{book_id}"
            )


            helpful = st.radio(
                "Was this recommendation helpful?",
                [
                    "Yes",
                    "No"
                ],
                horizontal=True,
                key=f"helpful_{book_id}"
            )


            if st.button(
                "Save feedback",
                key=f"save_{book_id}"
            ):

                try:

                    supabase.table(
                        "recommendation_feedback"
                    ).insert(
                        {
                            "user_id": user_id,

                            "source_book_id":
                                int(
                                    st.session_state
                                    .source_book_id
                                ),

                            "recommended_book_id":
                                book_id,

                            "rating":
                                int(rating),

                            "helpful":
                                helpful == "Yes"
                        }
                    ).execute()


                    st.success(
                        "Feedback saved!"
                    )


                except Exception as e:

                    st.error(
                        f"Could not save feedback: {e}"
                    )


# ============================================================
# MY RATINGS
# ============================================================

elif page == "⭐ My Ratings":

    st.title(
        "⭐ My Ratings"
    )


    try:

        ratings = (
            supabase
            .table(
                "book_ratings"
            )
            .select("*")
            .eq(
                "user_id",
                user_id
            )
            .order(
                "created_at",
                desc=True
            )
            .execute()
            .data
        )


        if ratings:

            ratings_df = pd.DataFrame(
                ratings
            )


            ratings_df = ratings_df.merge(
                books[
                    [
                        "book_id",
                        "title",
                        "author"
                    ]
                ],
                on="book_id",
                how="left"
            )


            st.dataframe(
                ratings_df[
                    [
                        "title",
                        "author",
                        "rating",
                        "created_at"
                    ]
                ],
                use_container_width=True
            )

        else:

            st.info(
                "You have not rated any books yet."
            )


    except Exception as e:

        st.error(
            f"Could not load your ratings: {e}"
        )
