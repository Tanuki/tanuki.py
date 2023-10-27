# standard / third party
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import uvicorn

# Configure sys path to include src dir for monkey patching
sys.path.append("../src")
# from monkey import Monkey

# local
from routers import router


app = FastAPI(
    docs_url="/docs",
    title="monkey-patch-apps",
)

app.include_router(router)

# origins = [CLIENT_URL]
# if APP_ENV != "production":
#     origins.append("https://demo.paperplane.ai")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["authorization", "x-app-version"],
# )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    # uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_config=log_config)
