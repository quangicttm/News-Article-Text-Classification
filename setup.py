import setuptools

setuptools.setup(
    name="News-Article-Topic-Text-Classification",
    version="0.0.2",
    author="James Stevenson",
    author_email="hi@jamesstevenson.me",
    description="A pre-trained text classification model for identifying the topic of a news article.",
    long_description="A pre-trained text classification model for identifying the topic of a news article.",
    long_description_content_type="text/markdown",
    url="https://github.com/user1342/News-Article-Text-Classification",
    packages=["news_classification"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['matplotlib==3.7.1', 
                      'seaborn==0.12.2', 
                      'feedparser==6.0.11', 
                      'pandas==1.5.3', 
                      'scikit-learn==1.2.2', 
                      'numpy==1.23.5', 
                      'newspaper3k==0.2.8', 
                      'pathlib==1.0.1'],
)
