from preprocessing_training_data import label_tweets, preprocessing
from pandas import DataFrame


# --- test labelling --- #

def test_init():
    path = '/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/'
    posi_filename = 'test_positive.json'  # around 400 tweets
    nega_filename = 'test_negative.json'  # around 400 tweets
    test_instance = label_tweets(path, posi_filename, nega_filename)

    assert isinstance(test_instance.tweet_df, DataFrame)
    assert isinstance(test_instance.posi_tweets_list, list)
    assert isinstance(test_instance.nega_tweets_list, list)
    assert isinstance(test_instance.posi_tweets_list[0], unicode)
    assert isinstance(test_instance.nega_tweets_list[0], unicode)
    assert len(test_instance.posi_tweets_list) > 0
    assert len(test_instance.nega_tweets_list) > 0

def test_label_the_tweets():
    path = '/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/'
    posi_filename = 'test_positive.json'  # around 400 tweets
    nega_filename = 'test_negative.json'  # around 400 tweets
    test_instance = label_tweets(path, posi_filename, nega_filename)
    tweets_list_mixed, combined_labels = test_instance.label_the_tweets()

    assert len(tweets_list_mixed) == len(test_instance.posi_tweets_list) + len(test_instance.nega_tweets_list)
    assert len(combined_labels) == len(tweets_list_mixed)

def test_take_only_the_pos_neg_tweets_and_labels():
    path = '/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/'
    posi_filename = 'test_positive.json'  # around 400 tweets
    nega_filename = 'test_negative.json'  # around 400 tweets
    test_instance = label_tweets(path, posi_filename, nega_filename)
    tweets_list_mixed, combined_labels = test_instance.label_the_tweets()
    pos_text_list, neg_text_list = test_instance.take_only_the_pos_neg_tweets_and_labels(tweets_list_mixed, combined_labels)

    assert len(pos_text_list)>0
    assert len(neg_text_list)>0

    assert isinstance(pos_text_list[0], unicode)
    assert isinstance(neg_text_list[0], unicode)

def test_main():
    path = '/Users/yibingyang/Documents/thesis_project_new/Data/Twitter/raw_data/'
    posi_filename = 'test_positive.json'  # around 400 tweets
    nega_filename = 'test_negative.json'  # around 400 tweets
    test_instance = label_tweets(path, posi_filename, nega_filename)
    tweets_list_mixed, combined_labels = test_instance.label_the_tweets()
    pos_text_list_processed, neg_text_list_processed = test_instance.take_only_the_pos_neg_tweets_and_labels(tweets_list_mixed, combined_labels)

    pos_text_list_main, neg_text_list_main = test_instance.main()

    assert sorted(pos_text_list_processed) == sorted(pos_text_list_main)
    assert sorted(neg_text_list_processed) == sorted(neg_text_list_main)


# --- test preprocessing --- #

def test_lowercase():
    after_labelled_text_list=[u'JUST WANT TO BE LOUD!!!']
    test_instance = preprocessing(after_labelled_text_list)
    test_text_list = test_instance.lowercase(test_instance.text_list)
    assert test_text_list == [u'just want to be loud!!!']

def test_convert_punc():
    after_labelled_text_list=[u'I ain\u2019t going to record\u2010\u2002']
    test_instance = preprocessing(after_labelled_text_list)
    test_text_list = test_instance.convert_punc(test_instance.text_list)
    assert test_text_list == [u"I ain't going to record- "]

def test_expand_contractions():
    after_labelled_text_list=[u"I'm not an easy man"]
    test_instance = preprocessing(after_labelled_text_list)
    test_text_list = test_instance.expand_contractions(test_instance.text_list)
    assert test_text_list == [u"I am not an easy man"]

def test_remove_punc_and_symbols():
    after_labelled_text_list=[ u'https://t.co/Obor3uOyJU' #url
                              ,u'@aabb_c so what' #@
                              ,u'<a>test html</a>' #html
                              ,u'numbers here 123' #number
                              ,u'string emoji:):-( :)' #string emoji
                              ,u'test\n\n' #\n
                              ,u'drop punctuations:!@#$%^&*()_-+[]|\/'":><.,;" ]
    test_instance = preprocessing(after_labelled_text_list)
    test_text_list = test_instance.remove_punc_and_symbols(test_instance.text_list)
    assert test_text_list == [ u''
                              ,u' so what'
                              ,u'test html'
                              ,u'numbers here '
                              ,u'string emoji '
                              ,u'test  '
                              ,u'drop punctuations' ]

def test_tokenize_and_stop_word_filter():
    text_list = [ u''
                 ,u' so what'
                 ,u'test html'
                 ,u'numbers here '
                 ,u'string emoji '
                 ,u'test  '
                 ,u'drop punctuations'
                 ,u'\U0001f612'
                 ]
    test_instance = preprocessing(text_list)
    test_text_list = test_instance.tokenize_and_stop_word_filter(test_instance.text_list)

    assert test_text_list == [[u'test', u'html'],
                              [u'numbers'],
                              [u'string', u'emoji'],
                              [u'test'],
                              [u'drop', u'punctuations'],
                             ]

def test_compressed_repetitive_words():
    text_list = [[u'soooo', u'gooood'], [u'ook' ,u'ooook']]
    test_instance = preprocessing(text_list)
    test_text_list = test_instance.compressed_repetitive_words(test_instance.text_list)
    assert test_text_list == [[u'soo', u'good'], [u'ook', u'ook']]

def test_stemming():
    text_list = [[u'dream', u'nasty', u'hopefully', u'better'], [u'rounded', u'ambiguous']]
    test_instance = preprocessing(text_list)
    test_text_list = test_instance.stemming(test_instance.text_list)
    assert test_text_list == [[u'dream', u'nasti', u'hope', u'better'], [u'round', u'ambigu']]

def test_lemmatizing(): #nothing really changes
    text_list = [[u'dream', u'nasty', u'hopefully', u'better'], [u'rounded', u'ambiguous']]
    test_instance = preprocessing(text_list)
    test_text_list = test_instance.lemmatizing(test_instance.text_list)
    assert test_text_list == [[u'dream', u'nasty', u'hopefully', u'better'], [u'rounded', u'ambiguous']]

def test_drop_duplicates():
    text_list = [[u'dream', u'land'],
                 [u'land', u'dream'],
                 [u'land', u'dream'],]
    test_instance = preprocessing(text_list)
    test_text_list = test_instance.drop_duplicates(test_instance.text_list)
    assert test_text_list == [[u'dream', u'land']]

def test_main():
    after_labelled_text_list = \
             [u'RT bitch'  # RT
            , u'RT bitch'  # dup
            , u'I will kill you \u203c\ufe0f \ud83d\ude14'  # emoji unicode
            , u'@wojespn: Bitches think cause you Avoid drama you pussy'  # @someone
            , u'SpaceJam 2 ain\u2019t going to record itself'  # ain't
            , u"I'm not an easy man"
            , u'keeps creating tsunamis https://t.\u2026'  # url & \u0206
            , u'PLEASE, HELP ME! https://t.co/Obor3uOyJU'  # url
            , u'JUST WANT TO BE LOUD!!!'  # upper letter
            , u'like you all arent tired of me'  # arent
            , u'test \n\n'  # \n
            , u'I SAW #GOT7'  ##GOT7
            , u'numbers here: 123'
            , u"I am a-word-with-puc two-link"  # a-b
            , u'string emoji :)'  # string emoji
            , u'drop punctuations :!@#$%^&*()_+[]|\/'":><.,;"  # puncs
            , u'repeatitive words appppppple goood'  # compressed words
         ]
    test_instance = preprocessing(after_labelled_text_list)
    test_text_list = test_instance.main()

    assert test_text_list ==[['bitch'],
                             ['kill'],
                             ['avoid', 'bitch', 'caus', 'drama', 'pussi', 'think'],
                             ['go', 'record', 'spacejam'],
                             ['easi', 'im', 'man'],
                             ['creat', 'keep', 'tsunami'],
                             ['help', 'pleas'],
                             ['loud', 'want'],
                             ['like', 'tire'],
                             ['test'],
                             ['got', 'saw'],
                             ['number'],
                             ['awordwithpuc', 'twolink'],
                             ['emoji', 'string'],
                             ['drop', 'punctuat'],
                             ['appl', 'good', 'repeatit', 'word']]
