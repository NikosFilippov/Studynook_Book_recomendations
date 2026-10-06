import os
import joblib
import pandas as pd
import streamlit as st

from scipy.sparse import load_npz
from supabase import create_client


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StudyNook Book Recommender",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# SUPABASE CONNECTION
# ============================================================

@st.cache_resource
def get_supabase():

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

    except Exception:
        st.error(
            """
            Supabase credentials are missing.

            Add SUPABASE_URL and SUPABASE_KEY
            to Streamlit Secrets.
            """
        )
        st.stop()

    return create_client(
        url,
        key
    )


supabase = get_supabase()


# ============================================================
# MODEL ARTIFACTS
# ============================================================

ARTIFACT_DIR = "artifacts"


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

    matrix_book_ids = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "matrix_book_ids.joblib"
        )
    )

    return (
        model,
        content_matrix,
        matrix_book_ids
    )


model, content_matrix, matrix_book_ids = load_model()


# ============================================================
# SUPABASE BOOK FUNCTIONS
# ============================================================

def search_editions(
    query,
    limit=30
):

    if not query or not query.strip():

        return pd.DataFrame()

    query = query.strip()

    response = (
        supabase
        .table("editions")
        .select(
            """
            edition_id,
            book_id,
            title,
            author,
            bookformat,
            isbn,
            isbn13,
            pages,
            rating,
            totalratings,
            img
            """
        )
        .or_(
            f"title.ilike.%{query}%,"
            f"author.ilike.%{query}%"
        )
        .limit(limit)
        .execute()
    )

    if not response.data:

        return pd.DataFrame()

    return pd.DataFrame(
        response.data
    )


def get_book(
    book_id
):

    response = (
        supabase
        .table("books")
        .select("*")
        .eq(
            "book_id",
            int(book_id)
        )
        .limit(1)
        .execute()
    )

    if not response.data:

        return None

    return response.data[0]


def get_books(
    book_ids
):

    if not book_ids:

        return pd.DataFrame()

    book_ids = [
        int(x)
        for x in book_ids
    ]

    response = (
        supabase
        .table("books")
        .select("*")
        .in_(
            "book_id",
            book_ids
        )
        .execute()
    )

    if not response.data:

        return pd.DataFrame()

    return pd.DataFrame(
        response.data
    )


def get_editions_for_book(
    book_id
):

    response = (
        supabase
        .table("editions")
        .select(
            """
            edition_id,
            book_id,
            title,
            author,
            bookformat,
            isbn,
            isbn13,
            pages,
            rating,
            totalratings,
            img
            """
        )
        .eq(
            "book_id",
            int(book_id)
        )
        .limit(50)
        .execute()
    )

    if not response.data:

        return pd.DataFrame()

    return pd.DataFrame(
        response.data
    )


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_books(
    book_id,
    n=10
):

    book_id = int(book_id)

    # --------------------------------------------------------
    # Find the row of this book inside the ML matrix
    # --------------------------------------------------------

    try:

        row_position = (
            matrix_book_ids.index(
                book_id
            )
        )

    except ValueError:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Get nearest neighbours
    # --------------------------------------------------------

    distances, indices = (
        model.kneighbors(
            content_matrix[
                row_position
            ],
            n_neighbors=min(
                n + 1,
                content_matrix.shape[0]
            )
        )
    )


    distances = distances[0]
    indices = indices[0]


    # --------------------------------------------------------
    # Convert matrix rows back to book IDs
    # --------------------------------------------------------

    recommended_ids = []

    similarities = []


    for distance, index in zip(
        distances,
        indices
    ):

        recommended_book_id = int(
            matrix_book_ids[index]
        )

        # Remove the selected book itself

        if recommended_book_id == book_id:
            continue

        recommended_ids.append(
            recommended_book_id
        )

        similarities.append(
            max(
                0,
                1 - float(distance)
            )
        )

        if len(recommended_ids) >= n:
            break


    # --------------------------------------------------------
    # Retrieve metadata from Supabase
    # --------------------------------------------------------

    recommendations = get_books(
        recommended_ids
    )


    if recommendations.empty:

        return recommendations


    # --------------------------------------------------------
    # Restore recommendation order
    # --------------------------------------------------------

    order = {
        book_id: position
        for position, book_id
        in enumerate(recommended_ids)
    }

    similarity_map = {
        book_id: similarity
        for book_id, similarity
        in zip(
            recommended_ids,
            similarities
        )
    }


    recommendations[
        "_order"
    ] = recommendations[
        "book_id"
    ].map(order)


    recommendations[
        "similarity"
    ] = recommendations[
        "book_id"
    ].map(similarity_map)


    recommendations = (
        recommendations
        .sort_values("_order")
        .drop(columns="_order")
    )


    return recommendations


# ============================================================
# SESSION STATE
# ============================================================

if "user" not in st.session_state:

    st.session_state.user = None


if "source_book_id" not in st.session_state:

    st.session_state.source_book_id = None


# ============================================================
# LOGIN
# ============================================================

def login():

    st.title(
        "📚 StudyNook Book Recommender"
    )

    st.write(
        """
        Find books you like and discover
        similar books using our recommendation model.
        """
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
                    """
                    Account created.

                    Check your email if email
                    confirmation is enabled.
                    """
                )


            except Exception as e:

                st.error(
                    f"Could not create account: {e}"
                )


# ============================================================
# SHOW LOGIN IF NOT LOGGED IN
# ============================================================

if st.session_state.user is None:

    login()

    st.stop()


user = st.session_state.user

user_id = user.id


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📚 StudyNook"
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


    st.divider()


    if st.button(
        "Logout"
    ):

        try:
            supabase.auth.sign_out()

        except Exception:
            pass


        st.session_state.user = None

        st.session_state.source_book_id = None

        st.rerun()


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "📚 StudyNook Book Recommender"
    )

    st.write(
        """
        Search for a book and receive recommendations
        based on content similarity and the Nearest
        Neighbours model.
        """
    )


    st.info(
        """
        You can select a specific edition of a book.

        The recommendation model works at the
        work level, so different editions of the
        same work lead to the same recommendation
        set.
        """
    )


    st.subheader(
        "How it works"
    )


    st.write(
        """
        1. Search for a book.

        2. Select the exact edition you want.

        3. The selected edition is connected to its
           underlying book/work.

        4. Nearest Neighbours finds similar books.

        5. You can rate the recommendations.
        """
    )


# ============================================================
# FIND A BOOK
# ============================================================

elif page == "🔎 Find a Book":

    st.title(
        "🔎 Find a Book"
    )


    st.write(
        """
        Search by title or author.
        """
    )


    query = st.text_input(
        "Book search",
        placeholder="Try: Harry Potter, Animal Farm, Dan Brown..."
    )


    if query:

        results = search_editions(
            query,
            limit=30
        )


        if results.empty:

            st.warning(
                "No books found."
            )


        else:

            st.write(
                f"Found {len(results)} results."
            )


            options = {}


            for _, row in results.iterrows():

                edition_id = int(
                    row["edition_id"]
                )


                title = str(
                    row["title"]
                )


                author = str(
                    row["author"]
                )


                bookformat = row.get(
                    "bookformat",
                    ""
                )


                isbn = row.get(
                    "isbn",
                    ""
                )


                if pd.isna(bookformat):

                    bookformat = ""


                if pd.isna(isbn):

                    isbn = ""


                label = (
                    f"{title} — {author}"
                )


                if str(
                    bookformat
                ).strip():

                    label += (
                        f" — {bookformat}"
                    )


                if str(
                    isbn
                ).strip():

                    label += (
                        f" — ISBN {isbn}"
                    )


                options[
                    edition_id
                ] = label


            selected_edition_id = (
                st.selectbox(
                    "Select the exact edition",
                    list(
                        options.keys()
                    ),
                    format_func=lambda x:
                        options[x]
                )
            )


            selected_edition = (
                results[
                    results[
                        "edition_id"
                    ]
                    == selected_edition_id
                ]
                .iloc[0]
            )


            st.divider()


            st.subheader(
                selected_edition[
                    "title"
                ]
            )


            st.write(
                f"**Author:** "
                f"{selected_edition['author']}"
            )


            if pd.notna(
                selected_edition.get(
                    "bookformat"
                )
            ):

                st.write(
                    f"**Format:** "
                    f"{selected_edition['bookformat']}"
                )


            if pd.notna(
                selected_edition.get(
                    "isbn"
                )
            ):

                st.write(
                    f"**ISBN:** "
                    f"{selected_edition['isbn']}"
                )


            if pd.notna(
                selected_edition.get(
                    "pages"
                )
            ):

                if float(
                    selected_edition["pages"]
                ) > 0:

                    st.write(
                        f"**Pages:** "
                        f"{int(selected_edition['pages'])}"
                    )


            if pd.notna(
                selected_edition.get(
                    "rating"
                )
            ):

                st.write(
                    f"**Goodreads rating:** "
                    f"{float(selected_edition['rating']):.2f}"
                )


            if st.button(
                "Use this book",
                type="primary"
            ):

                st.session_state[
                    "source_book_id"
                ] = int(
                    selected_edition[
                        "book_id"
                    ]
                )


                st.success(
                    "Book selected!"
                )


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


    if query:

        results = search_editions(
            query
        )


        if not results.empty:

            options = {}


            for _, row in results.iterrows():

                edition_id = int(
                    row["edition_id"]
                )


                label = (
                    f"{row['title']} — "
                    f"{row['author']}"
                )


                if pd.notna(
                    row.get("bookformat")
                ):

                    label += (
                        f" — {row['bookformat']}"
                    )


                options[
                    edition_id
                ] = label


            selected_edition_id = (
                st.selectbox(
                    "Select edition",
                    list(
                        options.keys()
                    ),
                    format_func=lambda x:
                        options[x]
                )
            )


            selected_edition = (
                results[
                    results[
                        "edition_id"
                    ]
                    == selected_edition_id
                ]
                .iloc[0]
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

                selected_book_id = int(
                    selected_edition[
                        "book_id"
                    ]
                )


                try:

                    # ----------------------------------------
                    # Reading history
                    # ----------------------------------------

                    (
                        supabase
                        .table(
                            "reading_history"
                        )
                        .upsert(
                            {
                                "user_id": user_id,
                                "book_id":
                                    selected_book_id,
                                "status": "read"
                            },
                            on_conflict=(
                                "user_id,book_id"
                            )
                        )
                        .execute()
                    )


                    # ----------------------------------------
                    # Rating
                    # ----------------------------------------

                    (
                        supabase
                        .table(
                            "book_ratings"
                        )
                        .upsert(
                            {
                                "user_id": user_id,
                                "book_id":
                                    selected_book_id,
                                "rating": int(
                                    rating
                                )
                            },
                            on_conflict=(
                                "user_id,book_id"
                            )
                        )
                        .execute()
                    )


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
            First go to "Find a Book"
            and select a book.
            """
        )

        st.stop()


    source_book_id = int(
        st.session_state.source_book_id
    )


    source = get_book(
        source_book_id
    )


    if source is None:

        st.error(
            "Selected book could not be found."
        )

        st.stop()


    st.write(
        "Recommendations based on:"
    )


    st.subheader(
        f"{source['title']} — "
        f"{source['author']}"
    )


    recommendations = recommend_books(
        source_book_id,
        n=10
    )


    if recommendations.empty:

        st.warning(
            "No recommendations found."
        )

        st.stop()


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


            if pd.notna(
                row.get("similarity")
            ):

                st.write(
                    f"**Similarity:** "
                    f"{float(row['similarity']):.1%}"
                )


            if row.get(
                "genre_list"
            ):

                genres = row[
                    "genre_list"
                ]

                if isinstance(
                    genres,
                    str
                ):

                    genres = genres.split(
                        "||"
                    )


                st.write(
                    "**Genres:** "
                    + ", ".join(
                        genres
                    )
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

                    (
                        supabase
                        .table(
                            "recommendation_feedback"
                        )
                        .insert(
                            {
                                "user_id":
                                    user_id,

                                "source_book_id":
                                    source_book_id,

                                "recommended_book_id":
                                    book_id,

                                "rating":
                                    int(rating),

                                "helpful":
                                    helpful == "Yes"
                            }
                        )
                        .execute()
                    )


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


        if not ratings:

            st.info(
                "You have not rated any books yet."
            )

            st.stop()


        ratings_df = pd.DataFrame(
            ratings
        )


        book_ids = (
            ratings_df[
                "book_id"
            ]
            .astype(int)
            .tolist()
        )


        books_df = get_books(
            book_ids
        )


        ratings_df = ratings_df.merge(
            books_df[
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


    except Exception as e:

        st.error(
            f"Could not load your ratings: {e}"
        )
