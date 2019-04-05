from preprocessing_test_data import read_and_cleaning, get_only_posi_nega_tweets_and_lables

url = '/Users/yibingyang/Documents/final_thesis_project/Data/Twitter/test_dataset/'
filename = 'test-SemEval2017-task4-test-dataset.txt'

# to be tested
def test_read_and_cleaning(url, filename):
    tweets, labels = read_and_cleaning(url, filename)
    assert isinstance(tweets, list)
    assert isinstance(labels, list)
    assert isinstance(tweets[0], str)
    assert labels in ['positive', 'negative']

def test_get_only_posi_nega_tweets_and_lables():
    tweets, labels = read_and_cleaning(url, filename)
    posi_tweets, nega_tweets = get_only_posi_nega_tweets_and_lables(tweets, labels)
    assert isinstance(posi_tweets, list)
    assert isinstance(nega_tweets, list)
    assert isinstance(posi_tweets[0], str)
    assert isinstance(nega_tweets[0], str)
