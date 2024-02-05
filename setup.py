from distutils.core import setup

from setuptools import find_packages

setup(
    name='tanuki.py',  # How you named your package folder (MyLib)
    packages=find_packages(),  # Chose the same as "name"
    version='0.2.0',
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='The easiest way to build scalable LLM-powered applications, which gets cheaper and faster over time.',
    # Give a short description about your library
    author='Jack Hopkins',  # Type in your name
    author_email='jack.hopkins@me.com',  # Type in your E-Mail
    url='https://github.com/Tanuki/tanuki.py',  # Provide either the link to your github or to your website
    download_url='https://github.com/Tanuki/tanuki.py/archive/v0.1.0.tar.gz',  # I explain this later on
    keywords=['python', 'ai', 'tdd', 'alignment', 'tanuki', 'distillation', 'pydantic', 'gpt-4', 'llm', 'chat-gpt', 'gpt-4-api', 'ai-functions'],  # Keywords that define your package best
    package_dir={'tanuki': './src/tanuki'},
    install_requires=[
        "appdirs~=1.4.4",
        "openai==1.3.5",
        "numpy>=1.17.3",
        "python-dotenv==1.0.0",
        "bitarray==2.8.2",
        "pydantic>1.0.0",
        "requests~=2.31.0"
    ],
    extras_require={
        'aws_bedrock': [
            "boto3==1.33.1",
            "botocore==1.33.1",
            # Add any additional dependencies for the optional feature here
        ],
        'together_ai': [
            "together==0.2.11",
            # Add any additional dependencies for the optional feature here
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which python versions that you want to support
    ],
)
