#!/usr/bin/env python
# md5: 8e71009e71cfea378a4b77d244aee94a
#!/usr/bin/env python
# coding: utf-8



import translationService
from core.vwconfig import ValidWordsConfig
from core.valid_background_words import ValidBackgroundWords
from core.valid_foreground_words import ValidForegroundWords
from core.service import TranslationRequest
from core.db import DbConnection

#import yaml
#import collections
import time
import utils.config
from prometheus_client import Counter
from core.timer import Timer

from getsecret import getsecret

#config = yaml.safe_load(open('../../config/generated/local/service.yaml', 'rt'))
config = utils.config.load_config('../../config/generated/local/service.yaml')
config['RDConfig']['pwd'] = getsecret('lilt_redis_password')



import sys
langpair_arg = sys.argv[1]
#src_lang = 'en'
#trg_lang = 'zh'
src_lang = langpair_arg[:2] #'en'
trg_lang = langpair_arg[2:] #'zh'



# service = TranslationUpdaterService(options.config, debug=options.debug, services=options.services,
#                                         healthcheck_server=hc_service)
# service.run()



connection = DbConnection(config)
#connection = DbConnection(config=None)
#connection = DbConnection(config={})
debug = False

valid_words_config = ValidWordsConfig(config)
valid_background_words = ValidBackgroundWords(valid_words_config)
#valid_background_words = ValidBackgroundWords(config=config)
valid_foreground_words = ValidForegroundWords(config=config, db=connection)
specific_language_pairs = {(src_lang, trg_lang)}
#specific_language_pairs = set()
#specific_language_pairs = None

inserted_items_counter = Counter(name='background_cache_inserted_items_counter',
                                                      documentation='Insertions into the background cache',
                                                      labelnames=['languages', 'service_name'])


popped_items_counter = Counter(name='background_cache_items_popped_counter',
                                                documentation='Background cache pops',
                                                labelnames=['languages', 'service_name'])

service = translationService.TranslationService(
  config,
  db=connection,
  debug=debug,
  cache_items_counters=(inserted_items_counter, popped_items_counter),
  valid_background_words=valid_background_words,
  valid_foreground_words=valid_foreground_words,
  specific_language_pairs=specific_language_pairs
)

service.setup()
service.load_language_pair(src_lang, trg_lang)

db_session = connection.session()
timer = Timer()



from ling.segmenter import get_segmenter

src_segmenter = get_segmenter(src_lang)
trg_segmenter = get_segmenter(trg_lang)

def translate(sentence, prefix=''):
  #sentence = '''Hello, how are you?'''
  request = TranslationRequest({
    "reqType": "Decode",
    "query": sentence, #"This varies from router to router.",
    "prefix": prefix,
    "srcLang": src_lang, #"en",
    "trgLang": trg_lang, #"de",
    "n": 1,
    #"model": 24,
    "model": -1,
    "modelVersion": -1,
    "docId": 411,
    "projectId": -1,
    "projectTags":False,
    "includeTranslationMemory": False,
    #"includeTranslationMemory": True,
    "includeMachineTranslation": True,
    "concordanceIsTarget": False,
    "autoSplit": True,
    #"timeThreshold": 0.001,
  })
  response = service.process(translation_request=request, db_session=db_session, timer=timer, log_vals={})
  translation = response[0]['translation'][0]
  return translation['targetWords'], translation['targetDelimiters'] # translation['target'],

#translate('hello world')



#print(time.perf_counter())



def get_word_end_indexes(tokens, delimiters):
  if len(delimiters) == 0:
    return []
  text_idx = len(delimiters[0])
  output = []
  for delimiter_idx,delimiter in list(enumerate(delimiters))[1:]:
    word_idx = delimiter_idx - 1
    word = tokens[word_idx]
    text_idx += len(word) + len(delimiter)
    output.append(text_idx)
  return output

def to_translated_text(words, delimiters):
  output = []
  if len(delimiters) == 0:
    return ""
  output.append(delimiters[0])
  for delimiter_idx,delimiter in list(enumerate(delimiters))[1:]:
    word_idx = delimiter_idx - 1
    word = words[word_idx]
    output.append(word)
    output.append(delimiter)
  return "".join(output)

#segmented = src_segmenter.segment(" ¿Hello hola?")
#segmented = src_segmenter.segment("¿Hello, are you there?") #src_segmenter.segment("¿Hello?")
#print(get_word_end_indexes(segmented.tokens, segmented.delimiters))

def get_remaining_words_and_delimiters_after_prefix(prefix, words, delimiters):
  if len(delimiters) == 0:
    return [], []
  if len(prefix) == 0:
    return words[:], delimiters[:]
  current_idx = len(delimiters[0])
  for delimiter_idx,delimiter in list(enumerate(delimiters))[1:]:
    word_idx = delimiter_idx - 1
    word = words[word_idx]
    current_idx += len(word)
    if current_idx >= len(prefix):
      remaining_words = words[word_idx + 1:]
      if len(remaining_words) > 0:
        return words[word_idx + 1:], delimiters[delimiter_idx:]
      else:
        return [], []
    else:
      current_idx += len(delimiter)
      if current_idx >= len(prefix):
        remaining_words = words[word_idx + 1:]
        if len(remaining_words) > 0:
          return remaining_words, [''] + delimiters[delimiter_idx + 1:]
        else:
          return [], []

# prefix = '过去'
# words = ['过去', '20', '年间', '，', '中国', '在线', '产业', '发展', '迅速', '，', '依靠', '中国', '庞大', '的', '市场', '，', '生产', '了', '与', '美国', '和', '欧洲', '互联网', '巨头', '，', '如', '阿里巴巴', '、', '腾讯', '、', '拜', '都', '、', '美川', '、', 'JD', '、', '拜丹斯', '等', '均衡', '的', '大量', '公司', '。']
# delimiters = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
# #target = to_translated_text(words, delimiters)
# #print(target)
# remaining_words, remaining_delimiters = get_remaining_words_and_delimiters_after_prefix(prefix, words, delimiters)
# print(len(remaining_words), len(remaining_delimiters))
# print(remaining_delimiters)

def print_simulation_results(output):
  num_total_predictions = len(output)
  num_correct_predictions = sum([1 for x in output if x[2] == x[3]])
  print('num correct predictions: ' + str(num_correct_predictions))
  print('num total predictions: ' + str(num_total_predictions))
  print('fraction of predictions correct: ' + str(num_correct_predictions / num_total_predictions))
  print('total amount of time spent for MT if recompute after every word: ' + str(sum([x[0] for x in output])))
  print('total amount of time spent for MT if only recompute after incorrect words: ' + str(sum([x[0] for x in output if x[1]])))


def simulate_words_that_need_to_be_typed(src_sentence, trg_sentence, verbose=True):
  #src_words = src_sentence.split(' ') # hack, replace with tokenizer
  src_sentence_segmented = src_segmenter.segment(src_sentence)
  trg_sentence_segmented = trg_segmenter.segment(trg_sentence)
  src_delimiters = src_sentence_segmented.delimiters
  trg_delimiters = trg_sentence_segmented.delimiters
  src_tokens = src_sentence_segmented.tokens
  trg_tokens = trg_sentence_segmented.tokens
  #print(src_tokens)
  #print(src_delimiters)
  #print(trg_tokens)
  #print(trg_delimiters)
  word_start_idx = 0
  to_print = []
  num_correct_predictions = 0
  num_total_predictions = 0
  current_time = time.perf_counter()
  #elapsed_time_info = []
  output = []
  is_mt_recompute_needed = True
  for word_end_idx in [0] + get_word_end_indexes(trg_tokens, trg_delimiters):
    #print()
    prefix = trg_sentence[:word_end_idx]
    to_print.append('=============')
    to_print.append('prefix: ' + prefix)
    before_time = time.perf_counter()
    predicted_words, predicted_delimiters = translate(src_sentence, prefix)
    after_time = time.perf_counter()
    elapsed_time = after_time - before_time
    #elapsed_time_info.append((elapsed_time, is_mt_recompute_needed))
    remaining_predicted_words, remaining_predicted_delimiters = get_remaining_words_and_delimiters_after_prefix(prefix, predicted_words, predicted_delimiters)
    remaining_actual_words, remaining_actual_delimiters = get_remaining_words_and_delimiters_after_prefix(prefix, trg_tokens, trg_delimiters)
    to_print.append('remaining prediction: ' + to_translated_text(remaining_predicted_words, remaining_predicted_delimiters))
    to_print.append('remaining actual: ' + to_translated_text(remaining_actual_words, remaining_actual_delimiters))
    to_print.append('time elapsed for computing MT: ' + str(elapsed_time))
    to_print.append('was MT recompute needed: ' + str(is_mt_recompute_needed))
    if len(remaining_actual_words) > 0:
      next_actual_word = remaining_actual_words[0]
      next_predicted_word = ''
      if len(remaining_predicted_words) > 0:
        next_predicted_word = remaining_predicted_words[0]
      correct = next_actual_word == next_predicted_word
      is_next_actual_word_anywhere_in_suffix = next_actual_word in remaining_predicted_words
      num_total_predictions += 1
      if correct:
        output.append([elapsed_time, is_mt_recompute_needed, next_actual_word, remaining_predicted_words, remaining_predicted_delimiters])
        #output.append([elapsed_time, is_mt_recompute_needed, next_actual_word, next_predicted_word, is_next_actual_word_anywhere_in_suffix])
        is_mt_recompute_needed = False
        num_correct_predictions += 1
        to_print.append('correctly predicted: ' + next_actual_word)
      else:
        output.append([elapsed_time, is_mt_recompute_needed, next_actual_word, remaining_predicted_words, remaining_predicted_delimiters])
        #output.append([elapsed_time, is_mt_recompute_needed, next_actual_word, next_predicted_word, is_next_actual_word_anywhere_in_suffix])
        is_mt_recompute_needed = True
        to_print.append('incorrectly predicted: ' + next_predicted_word + ' actual word is: ' + next_actual_word)
    #to_print.append('remaining predicted words: ' + str(remaining_predicted_words))
    #to_print.append(remaining_predicted_delimiters)
    #print(prefix)
    #print(translate(src_sentence, prefix))
  if verbose:
    for x in to_print:
      print(x)
    print_simulation_results(output)
  return output
  #return num_correct_predictions, num_total_predictions, elapsed_time_info

#simulate_words_that_need_to_be_typed('Hello, how are you?', '你好吗')
# simulate_words_that_need_to_be_typed(
#   "In the past 20 years, China's online industry has rapidly developed, relying on the massive Chinese market, it has produced massive companies that are evenly matched with American and European internet giants, such as Alibaba, Tencent, Baidu, Meituan, JD, ByteDance, and more.",
#   '过去20年间，中国互联网企业快速发展，依托庞大的中国市场，成长出阿里巴巴、腾讯、百度、美团、京东、字节跳动等可以匹敌欧美互联网巨头的大公司。'
# )



# import msgpack
# import rocksdb
# cache_db = rocksdb.DB('simulation_cache/' + langpair_arg, rocksdb.Options(create_if_missing=True))
# translation_cache_db = rocksdb.DB('translation_cache/' + langpair_arg, rocksdb.Options(create_if_missing=True))



import msgpack
import lmdb

cache_db = lmdb.open('simulation_cache/' + langpair_arg, map_size=34359738368, sync=True, map_async=False, writemap=False)
translation_cache_db = lmdb.open('translation_cache/' + langpair_arg, map_size=34359738368, sync=True, map_async=False, writemap=False)



def simulate_words_that_need_to_be_typed_cached(src_sentence, trg_sentence):
  key = msgpack.dumps([src_sentence, trg_sentence])
  txn = cache_db.begin()
  cached_result = txn.get(key)
  txn = None
  #cached_result = cache_db.get(key)
  if cached_result is None:
    txn = cache_db.begin(write=True)
    cached_result = simulate_words_that_need_to_be_typed(src_sentence, trg_sentence)
    txn.put(key, msgpack.dumps(cached_result))
    txn.commit()
    txn = None
    #cache_db.put(key, msgpack.dumps(cached_result))
    return cached_result
  else:
    return msgpack.loads(cached_result, raw=False, strict_map_key=False)

def get_translation_cached(src_sentence):
  src_sentence_key = msgpack.dumps(src_sentence)
  txn = translation_cache_db.begin()
  cached_translation = txn.get(src_sentence_key)
  txn = None
  if cached_translation is not None:
    return msgpack.loads(cached_translation, raw=False, strict_map_key=False)
  predicted_words, predicted_delimiters = translate(src_sentence, '')
  txn = translation_cache_db.begin(write=True)
  txn.put(src_sentence_key, msgpack.dumps([predicted_words, predicted_delimiters]))
  txn.commit()
  txn = None
  return [predicted_words, predicted_delimiters]

# results = simulate_words_that_need_to_be_typed_cached(
#   "In the past 20 years, China's online industry has rapidly developed, relying on the massive Chinese market, it has produced massive companies that are evenly matched with American and European internet giants, such as Alibaba, Tencent, Baidu, Meituan, JD, ByteDance, and more.",
#   '过去20年间，中国互联网企业快速发展，依托庞大的中国市场，成长出阿里巴巴、腾讯、百度、美团、京东、字节跳动等可以匹敌欧美互联网巨头的大公司。'
# )
# print_simulation_results(results)



# # noexport
# import glob
# langpairs = set()
# source_files = glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/source.txt')
# for source_file in source_files:
#   langpair = source_file.replace('/Users/geza/intel_evaluation/GCP_translate_test/', '').split('/', 1)[0]
#   langpairs.add(langpair)
# print(sorted(list(langpairs)))



#glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/**')



import glob

source_files = glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/source.txt')
target_files = glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/target.txt')
target_files_set = set(target_files)
for source_file_idx,source_file in enumerate(source_files):
  print(source_file_idx + 1, '/', len(source_files), source_file)
  target_file = source_file.replace('source.txt', 'target.txt')
  if target_file not in target_files_set:
    print('missing target file for: ' + source_file)
    continue
  source_lines = open(source_file).readlines()
  target_lines = open(target_file).readlines()
  if len(source_lines) != len(target_lines):
    print('source line length', len(source_lines), 'does not match target line length', len(target_lines))
    continue
  source_lines = [x.strip() for x in source_lines]
  target_lines = [x.strip() for x in target_lines]
  langpair = source_file.replace('/Users/geza/intel_evaluation/GCP_translate_test/', '').split('/', 1)[0]
  if langpair != langpair_arg:
    continue
  #print(langpair)
  src_lang = langpair[0:2]
  trg_lang = langpair[2:]
  for src_sentence,trg_sentence in zip(source_lines, target_lines):
    try:
      get_translation_cached(src_sentence)
    except:
      continue
    try:
      simulate_words_that_need_to_be_typed_cached(src_sentence, trg_sentence)
    except:
      continue



cache_db_txn = cache_db.begin()
translation_cache_db_txn = translation_cache_db.begin()

source_file_to_google = {}
source_files = glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/source.txt')
for source_file in source_files:
  google_translate_file = glob.glob(source_file.replace('source.txt', 'results/*.txt'))[0]
  source_file_to_google[source_file] = google_translate_file

output_list = []
#cache_db = langpair_to_cache_db[langpair_arg]
source_files = glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/source.txt')
target_files = glob.glob('/Users/geza/intel_evaluation/GCP_translate_test/**/**/**/target.txt')
target_files_set = set(target_files)
for source_file in source_files:
  target_file = source_file.replace('source.txt', 'target.txt')
  if target_file not in target_files_set:
    print('missing target file for: ' + source_file)
    continue
  google_translate_file = source_file_to_google[source_file]
  google_translate_lines = open(google_translate_file).readlines()
  source_lines = open(source_file).readlines()
  target_lines = open(target_file).readlines()
  if len(source_lines) != len(target_lines):
    print('source line length', len(source_lines), 'does not match target line length', len(target_lines))
    continue
  source_lines = [x.strip() for x in source_lines]
  target_lines = [x.strip() for x in target_lines]
  langpair = source_file.replace('/Users/geza/intel_evaluation/GCP_translate_test/', '').split('/', 1)[0]
  if langpair != langpair_arg:
    continue
  #print(langpair)
  src_lang = langpair[0:2]
  trg_lang = langpair[2:]
  for src_sentence,trg_sentence,google_translate in zip(source_lines, target_lines, google_translate_lines):
    key = msgpack.dumps([src_sentence, trg_sentence])
    cached_result = cache_db_txn.get(key)
    if cached_result is not None:
      output = msgpack.loads(cached_result, raw=False, strict_map_key=False)
      output_list.append([src_sentence, trg_sentence, output, google_translate])

msgpack.dump(output_list, open('simulation_output_list/' + langpair_arg + '.msgpack', 'wb'))

translation_dict = {}
for src_sentence,trg_sentence,output,google_translate in output_list:
  key = msgpack.dumps(src_sentence)
  cached_result = translation_cache_db_txn.get(key)
  if cached_result is not None:
    word_list,delimiter_list = msgpack.loads(cached_result)
    translation_dict[src_sentence] = [word_list, delimiter_list]

msgpack.dump(translation_dict, open('translation_dict/' + langpair_arg + '.msgpack', 'wb'))



import sys
sys.exit(0)




















# from utils.run_mode_types import RunMode, RunModeTypes

# RunModeTypes.is_service_mode(service._model_manager.run_mode)



# service._config[service._service_configuration_key()]['graphs']



# service._graph_configs



# from core.graphConfigs import GraphConfigs

# service._graph_configs = GraphConfigs(service._config[service._service_configuration_key()]['graphs'],
#                                            None)



# from core.graphConfigs import GraphConfigs

# service._graph_configs = GraphConfigs(service._config[service._service_configuration_key()]['graphs'],
#                                            specific_language_pairs)



#specific_language_pairs



# {'en', 'de'}.intersection([('en', 'de')])



# {('en', 'de')}.intersection([('en', 'de')])



# from core.langConfig import LangConfig

# for graph, conf in service._config[service._service_configuration_key()]['graphs'].items():
#   #print(conf)
#   lang_config = LangConfig(conf)
#   print(lang_config.language_pairs())
#   print(specific_language_pairs)
#   print(specific_language_pairs.intersection(lang_config.language_pairs()))



# lang_config.language_pairs()



# specific_language_pairs.intersection(lang_config.language_pairs())



# service._graph_configs



# service._graph_configs.lang_pair_to_graph



# service._model_manager.get_graph_deployment(('en', 'de'), blocking=True)



# service._model_manager._lang_pair_to_graph



# service._model_manager.language_pair_needs_loading(('en', 'de'))



# service.load_language_pair('en', 'de')



#service._setup_segmenters()



#from vocabulary import VocabularyConfig, Vocabulary

#service.vocab_cfg = VocabularyConfig(service._config)





