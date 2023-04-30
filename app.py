from flask import Flask, request, jsonify
from paraphraser import Paraphraser

app = Flask(__name__)


@app.route('/paraphrase', methods=['GET'])
def paraphrase():
    tree = request.args.get('tree', None)
    limit = int(request.args.get('limit', 0))

    if not tree:
        return 'Please provide a syntax tree to paraphrase', 400

    paraphraser = Paraphraser(tree, limit)

    result = paraphraser.paraphrase(['Noun phrases'])
    if isinstance(result, str):
        return result, 400

    paraphrased_trees = result
    response = {
        'paraphrases': [
            {'tree': str(tree)} for tree in paraphrased_trees
        ]
    }

    return jsonify(response), 200
