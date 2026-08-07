# Production-Grade RAG Refactor Guide (Simple Version)

This guide is written for you to follow step by step. The idea is to turn your current script-style code into a cleaner, production-style structure using classes, methods, and `self`.

---

## 1. Why this structure is better

Right now, your code may work, but it can become hard to manage when it grows.

A production-style project should be:
- easy to read
- easy to test
- easy to extend
- easy to maintain

The best way to do that is to group code into classes.

---

## 2. What is a class and why use `self`?

A class is a blueprint for an object.

Example:

```python
class DocumentProcessor:
    def __init__(self, settings):
        self.settings = settings

    def process(self, file_path):
        print(f"Processing: {file_path}")
```

### What happens here?
- `__init__` runs when you create an object.
- `self.settings` stores the settings inside the object.
- later, when you call `process()`, the method can use `self.settings`.

### Why `self` matters
`self` lets the object remember data between method calls.

Example:

```python
processor = DocumentProcessor(settings)
processor.process("book.pdf")
```

The object keeps the settings inside itself so the methods can use them later.

---

## 3. The simple production structure you should follow

Use this structure:

```text
app.py              -> starts the whole app
config.py           -> settings and environment values
services/
  pdf_processor.py  -> read and process PDFs
  embedding.py      -> create embeddings
  vector_store.py  -> store and retrieve vectors
  retriever.py     -> search relevant docs
  chat_service.py  -> build answer flow
prompts/            -> prompt templates
tests/              -> test your code
```

---

## 4. Step-by-step refactor plan

### Step 1: Move settings into a class

Keep your configuration in one place.

Example:

```python
class AppSettings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.collection_name = os.getenv("COLLECTION_NAME", "book_store")
```

### What happens here?
- the settings object is created once
- all other classes can use it
- no need to pass many variables around manually

---

### Step 2: Create a PDF processor class

This class should read and prepare your documents.

```python
class PDFProcessor:
    def __init__(self, settings):
        self.settings = settings

    def load_pdf(self, file_path):
        print(f"Loading PDF: {file_path}")
        return {"path": file_path}
```

### What happens here?
- `self.settings` stores config
- `load_pdf()` uses that config when needed
- the class becomes reusable

---

### Step 3: Create an embedding class

This class should create vector embeddings.

```python
class EmbeddingService:
    def __init__(self, settings):
        self.settings = settings

    def create_embedding(self, text):
        print(f"Creating embedding for: {text[:30]}")
        return [0.1, 0.2, 0.3]
```

### What happens here?
- you keep embedding logic in one place
- later you can swap model providers without changing everything else

---

### Step 4: Create a vector store class

This class handles storage and search.

```python
class VectorStoreService:
    def __init__(self, settings):
        self.settings = settings

    def add_documents(self, documents):
        print(f"Adding {len(documents)} documents")

    def similarity_search(self, query):
        print(f"Searching for: {query}")
        return ["doc1", "doc2"]
```

### What happens here?
- your retrieval logic stays isolated
- the rest of the app does not need to know how the database works

---

### Step 5: Create a retriever class

This class combines the vector store and the query logic.

```python
class RetrieverService:
    def __init__(self, vector_store, settings):
        self.vector_store = vector_store
        self.settings = settings

    def retrieve(self, query):
        return self.vector_store.similarity_search(query)
```

### What happens here?
- `self.vector_store` stores the vector store object
- `self.settings` stores the config
- the method uses them without needing global variables

---

### Step 6: Create one main app class

This class orchestrates everything.

```python
class RAGApp:
    def __init__(self, settings):
        self.settings = settings
        self.pdf_processor = PDFProcessor(settings)
        self.embedding_service = EmbeddingService(settings)
        self.vector_store = VectorStoreService(settings)
        self.retriever = RetrieverService(self.vector_store, settings)

    def run(self, query):
        docs = self.retriever.retrieve(query)
        return docs
```

### What happens here?
- the app creates all components once
- each class has one job
- the flow is easy to follow

---

## 5. Simple rule for using classes

When you build a class, always ask:

1. What data should this class remember?
2. What actions should this class perform?
3. What should be stored as `self`?

Example:

```python
class ChatService:
    def __init__(self, settings):
        self.settings = settings

    def answer(self, question):
        return f"Answering: {question}"
```

Here:
- `self.settings` is data the class stores
- `answer()` is an action the class performs

---

## 6. What you should avoid

Avoid this style:

```python
# bad style
x = None

def do_work():
    global x
    x = something
```

This becomes hard to manage.

Prefer this style:

```python
class Worker:
    def __init__(self):
        self.x = None

    def do_work(self):
        self.x = something
```

---

## 7. How to migrate your current files

You can gradually move your current modules into this pattern:

- keep [config.py](config.py) as the settings class
- move PDF logic into a processor class
- move embedding logic into an embedding service class
- move vector DB access into a vector store class
- create one orchestrator class to connect everything

---

## 8. A very simple checklist

Use this checklist every time you refactor:

- [ ] Put config in one class
- [ ] Give each class one responsibility
- [ ] Store shared data in `self`
- [ ] Keep methods small
- [ ] Avoid global variables
- [ ] Make one main orchestrator class
- [ ] Add tests for each class

---

## 9. Final goal

Your final code should look like this:

```python
settings = AppSettings()
app = RAGApp(settings)
result = app.run("What is this book about?")
```

That means:
- settings are loaded once
- the app is organized clearly
- each part is easier to manage
- future changes become much simpler

---

## 10. Recommended next step

Start by refactoring only one file at a time:

1. config
2. pdf processor
3. vector store
4. retriever
5. app orchestrator

Do not try to rewrite everything at once.

If you follow this guide slowly, your project will become much more production-ready.
