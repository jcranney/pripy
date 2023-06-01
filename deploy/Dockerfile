FROM python:3
WORKDIR /app
COPY _deploy.sh deploy.sh
RUN pip install build twine
CMD ["bash","./deploy.sh"]