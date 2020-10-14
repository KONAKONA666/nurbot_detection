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

print('kek')

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
	tokens = make_word2vec_vector_cnn(cleaner(data['text']))
	#print(tokens)
	return web.json_response({'tokens': tokens})


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
