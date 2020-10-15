from aiohttp import web
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

from gensim.models import Word2Vec
import aiohttp_cors

print('kek')
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)
w2v_model = Word2Vec.load("model")
max_sen_len = 748
padding_idx = 0

bots = ['http://www.youtube.com/channel/UCtp7rumEMZJlCq4Qh47agwQ',
 'http://www.youtube.com/channel/UCtKJJFxdONEjwGC9Pd_sGRA',
 'http://www.youtube.com/channel/UCfWCa1iMecNT1eDzt-CdxSg',
 'http://www.youtube.com/channel/UCEXDXQOgouCL1GwZyppFCQw',
 'http://www.youtube.com/channel/UCiN_404Op87L_EmCt09sNvg',
 'http://www.youtube.com/channel/UClEymv8p6zguWZWbRYfXmAg',
 'http://www.youtube.com/channel/UC2m2WZVpWFEvm2H68ICHEUA',
 'http://www.youtube.com/channel/UCRbI93i8L9q_ufXo9t8zL5Q',
 'http://www.youtube.com/channel/UCtOy6oYbhDzvOa276utAYXg',
 'http://www.youtube.com/channel/UC2FmTNeRTCymR0xNxfS3xWg',
 'http://www.youtube.com/channel/UCMfEIiiQEKQ_7seZlp6QukQ',
 'http://www.youtube.com/channel/UCGoABZHfzZLMx3ZkKtxJKEw',
 'http://www.youtube.com/channel/UC1xgD1aZ44OiRbjVRj6Rp0A',
 'http://www.youtube.com/channel/UCWHVUZNTQBRji_ftdZgL6hA',
 'http://www.youtube.com/channel/UCXK4hJOx4HnpfmK0FSzGMbg',
 'http://www.youtube.com/channel/UC2jbtjKl5Ovv86vaBewzmxg',
 'http://www.youtube.com/channel/UCNRWnqERq2ujqM7PUfQx2yg',
 'http://www.youtube.com/channel/UCEDkTyOOE3hCoPxqPGJYAHA',
 'http://www.youtube.com/channel/UCHgi4lteIgX_SHMYYvgVhCw',
 'http://www.youtube.com/channel/UC9MYdasd7tpRjSB3isYnkbg',
 'http://www.youtube.com/channel/UCXFWfKjh4YBK3jsgUjfKcEw',
 'http://www.youtube.com/channel/UCZA-vtsTJoBBJwXGZ-h8smA',
 'http://www.youtube.com/channel/UC0lpUAcbitgJqrjSvyvo8Yw',
 'http://www.youtube.com/channel/UC4esSaQ_KCIXHEwobjOGYhw',
 'http://www.youtube.com/channel/UCsriqoQ3QUomaqDK466gSgw',
 'http://www.youtube.com/channel/UC5Nkd8OPLJXDzD9DvOW4j5w',
 'http://www.youtube.com/channel/UCfaS7SCT5wEL6jP43dXa4Zw',
 'http://www.youtube.com/channel/UCjmkH6WAoSO7UV0buq_hD5w',
 'http://www.youtube.com/channel/UCV8JjJ_1xUioVXFfJmu5V1g',
 'http://www.youtube.com/channel/UCMXwF-g3bM0-BTS4s0QMB-Q',
 'http://www.youtube.com/channel/UCPwICSuqaDgPNAFsaoUxTqg',
 'http://www.youtube.com/channel/UC6JDzjUgBQ_ocIrvBcnFxDA',
 'http://www.youtube.com/channel/UCXSZwhPgXMwbsvZeMmz-nqg',
 'http://www.youtube.com/channel/UCbVpD3ObPvIyPysF11jDc9Q',
 'http://www.youtube.com/channel/UCR6M5XnQhh0Q3FK1kPA3lRw',
 'http://www.youtube.com/channel/UCuz42mPwx0hd-hD6iFw9bLg',
 'http://www.youtube.com/channel/UCeneTWEPGCIZxzCQPB6HkSw',
 'http://www.youtube.com/channel/UCP5VuZuLlSm_jRX08zlKwPQ',
 'http://www.youtube.com/channel/UCSLiv-CTNfhE3PqwNK71cKw',
 'http://www.youtube.com/channel/UCItGhuijDgHCEennYNKcaDg',
 'http://www.youtube.com/channel/UCp0rAU1yQ9BLeTRQHVve9bQ',
 'http://www.youtube.com/channel/UCOOjlEbulQg_DR59WYmYk5w',
 'http://www.youtube.com/channel/UCiNiUYTDUqUrItvdEwWibXA',
 'http://www.youtube.com/channel/UCQnW-9U23kiyhypBEVlRRPg',
 'http://www.youtube.com/channel/UCV-LyhMop-VyrNTwi2cTOFQ',
 'http://www.youtube.com/channel/UCiMK2wiHlpF4ne1pksttxtQ',
 'http://www.youtube.com/channel/UCcxPtdj8YReN3ZZfJkTkr7w',
 'http://www.youtube.com/channel/UCIK_yVcCFq_noAzmW3MBloQ',
 'http://www.youtube.com/channel/UCjmS5fVybsLSA871OX5CQkA',
 'http://www.youtube.com/channel/UCUtvyDTGXnPFWQPc6QYVROw',
 'http://www.youtube.com/channel/UC50l3kPm1HVtPAOUSw6XGjg',
 'http://www.youtube.com/channel/UC9bFHLYV6px2_DvWQhkzAKQ',
 'http://www.youtube.com/channel/UCtMKx8F4H4Z9FA-j2pgNJEA',
 'http://www.youtube.com/channel/UCIgQtZBO1BjtO9LU93EMVIg',
 'http://www.youtube.com/channel/UCaBUZYY4gh7IhRoTrOjK3cQ',
 'http://www.youtube.com/channel/UCSbWTHVuHLvZOzCtSJygi7w',
 'http://www.youtube.com/channel/UCgwKh_eMcJ9uRKLigAXLoQw',
 'http://www.youtube.com/channel/UCmNywZRU6fsoxNmrkWbwSYw',
 'http://www.youtube.com/channel/UCK1kBmiKpe6SfSPARDAE4Kw',
 'http://www.youtube.com/channel/UC38BZbv6e9_Qu60yzr9U8vA',
 'http://www.youtube.com/channel/UCutirAfbXoeNdLTqcBN980Q',
 'http://www.youtube.com/channel/UCsldqMfEQop9JSW2pT466Qw',
 'http://www.youtube.com/channel/UCx46Y_3l3ma0fOKnc9Xytsw',
 'http://www.youtube.com/channel/UCyTtpyCYmHHan-3DitoIXgg',
 'http://www.youtube.com/channel/UCkXDzsW75GcBNuTnQjQGkZQ',
 'http://www.youtube.com/channel/UCdA8cLm8visK04DYZ2nClQQ',
 'http://www.youtube.com/channel/UCR2xdBUFYfZmC75SRqU7HqQ',
 'http://www.youtube.com/channel/UC47RrFXQ7O0rRSi5BfPlpQQ',
 'http://www.youtube.com/channel/UCWt_kns-QnOJjJJwulkoidQ',
 'http://www.youtube.com/channel/UCuVNx_XxUoowfJ-wbyLMZ8A',
 'http://www.youtube.com/channel/UChzNPclex_5OjY5qGNNmpuA',
 'http://www.youtube.com/channel/UCCJ_bjNiTWavjngsYQX0YTA',
 'http://www.youtube.com/channel/UCsj1x6HtMjQNgxi0DJVy5aA',
 'http://www.youtube.com/channel/UCohtLJHlaGIcEMhMWr7dyjA',
 'http://www.youtube.com/channel/UCsEA-AC9KzId1VMKyGbVM3w',
 'http://www.youtube.com/channel/UCwXJ_9vk_fXkEUC0-d4SORQ',
 'http://www.youtube.com/channel/UC8N4BATvuBFqMRpfhesonhw',
 'http://www.youtube.com/channel/UCtKYbMXoSJ1bwOBA0r-4iMg',
 'http://www.youtube.com/channel/UCtlqtyG96XROXrpBe1iWvtA',
 'http://www.youtube.com/channel/UCBoN7k5-WPN17eU7Lwygo8w',
 'http://www.youtube.com/channel/UCW51OmYZLS0PYGXAd1Pgfbw',
 'http://www.youtube.com/channel/UCxRyhZDU47C2QeG33eQKyRA',
 'http://www.youtube.com/channel/UCQG9fgKS8QIefzPUEZwwAxg',
 'http://www.youtube.com/channel/UC-6UV4s1y-q7ZEPaJ8xaasA',
 'http://www.youtube.com/channel/UCCcP6-Kz88Msmp9V-eW6TFw',
 'http://www.youtube.com/channel/UC3QiJYWgTxw08WisyZwPRpQ',
 'http://www.youtube.com/channel/UCxGQaumjxweJA07wdO7wGhQ',
 'http://www.youtube.com/channel/UCTdsAg9XJxEAEYf7PwJ6Now',
 'http://www.youtube.com/channel/UCVSUafA6NUmjSbCezpfQnQg',
 'http://www.youtube.com/channel/UCcNRRWDiJY2CL3ZML2VClTw',
 'http://www.youtube.com/channel/UCEKOEIXXloL_POjbLbWYM0Q',
 'http://www.youtube.com/channel/UCjFLZ2yqxWUAkNsuOVRywpg',
 'http://www.youtube.com/channel/UCbplRcK8fAqMC20p_g2JEAA',
 'http://www.youtube.com/channel/UC3G22NfCADzAfp0ihG3B23w',
 'http://www.youtube.com/channel/UCm3hdypqwJr-Ry3gWnrgjHw',
 'http://www.youtube.com/channel/UCKzS1j4aWQGzcxtoablbp-A',
 'http://www.youtube.com/channel/UC2WPBTQF9YmIyjylc_kcuRw',
 'http://www.youtube.com/channel/UC7Yb3K-jgDMet3_r9p0D7_w',
 'http://www.youtube.com/channel/UC_83ovguYTj7ooAKf7LMg9A',
 'http://www.youtube.com/channel/UC664GJp6RFqtqLTxiOZujgA'
]
print("kek")

def is_in_botdf(url):
	url_filtred = url.replace('https', 'http')
	return url_filtred in bots

def cleaner(text):
 # out = []
	doc = Doc(text)
	doc.segment(segmenter)
	doc.tag_morph(morph_tagger)
	for token in doc.tokens:
		token.lemmatize(morph_vocab)
	out = [token.lemma for token in doc.tokens if token.pos != 'PUNCT']
	if len(out) > 2:
		return out


def make_word2vec_vector_cnn(sentence):
	padded_X = [padding_idx for i in range(max_sen_len)]
	i = 0
	for word in sentence:
		if word not in w2v_model.wv.vocab:
			padded_X[i] = 0
	       # print(word)
		else:
			padded_X[i] = w2v_model.wv.vocab[word].index
		i += 1
	return padded_X


async def get_tokens(request):
	data = await request.json()
	user_url = data['user_url']
	is_in_bots = False
	if is_in_botdf(user_url):
		tokens = [0]
		is_in_bots = True
	else:
		tokens = make_word2vec_vector_cnn(cleaner(data['text']))
	return web.json_response({'is_in_bots': is_in_bots, 'tokens': tokens})


if __name__ == '__main__':
	app = web.Application()
	cors = aiohttp_cors.setup(app)
	resource = cors.add(app.router.add_resource("/get_tokens"))
	route = cors.add(
    	resource.add_route("POST", get_tokens), {
        	"*": aiohttp_cors.ResourceOptions(
            	allow_credentials=True,
            	expose_headers='*',
            	allow_headers='*',
        	)
    })

	#app.add_routes([web.post('/get_tokens', get_tokens)])
	web.run_app(app)
