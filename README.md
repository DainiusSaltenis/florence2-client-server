# florence2-client-server readme

The repo is for Florence-2 inference. It is composed of client and server parts. 

Server part is implemented as a FastAPI REST API uvicorn app.

To run:

1. clone the repo
2. `cd` to `./florence2_api` and launch the uvicorn webserver app `uvicorn app:app --host 0.0.0.0 --port 8080 --reload`
3. (does not work yet) pip install the client part of the repository for use as a package using `pip install git+https://github.com/DainiusSaltenis/florence2-client-server.git` 
4. (alternative to #3) due to package conflict issues, instead build the conda environment locally from `./resources/envs/environment_dev.yml` (or build your own) and install pip package locally with `pip install -e /home/[where_your_cloned_repo_is_located]/florence2-client-server` 
5. import appropriate client classes for the webserver API use as `import florence2_client`
6. for more guidance of use and implemented code, please see the notebook at `./resources/notebooks/florence2-client-example-use.ipynb`
7. enjoy (if it works well)