import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile

from .util.paths import get_binary
from .metasentence import MetaSentence
from .resources import Resources

MKGRAPH_PATH = get_binary("ext/m3")

# [oov] no longer in words.txt
OOV_TERM = '<unk>'

def make_bigram_lm_fst(word_sequences, **kwargs):
    '''
    Use the given token sequence to make a bigram language model
    in OpenFST plain text format.

    When the "conservative" flag is set, an [oov] is interleaved
    between successive words.

    When the "disfluency" flag is set, a small set of disfluencies is
    interleaved between successive words

    `Word sequence` is a list of lists, each valid as a start
    '''

    if len(word_sequences) == 0 or type(word_sequences[0]) != list:
        word_sequences = [word_sequences]

    conservative = kwargs['conservative'] if 'conservative' in kwargs else False
    disfluency = kwargs['disfluency'] if 'disfluency' in kwargs else False
    disfluencies = kwargs['disfluencies'] if 'disfluencies' in kwargs else []

    bigrams = {OOV_TERM: set([OOV_TERM])}

    for word_sequence in word_sequences:
        if len(word_sequence) == 0:
            continue

        prev_word = word_sequence[0]
        bigrams[OOV_TERM].add(prev_word) # valid start (?)

        if disfluency:
            bigrams[OOV_TERM].update(disfluencies)

            for dis in disfluencies:
                bigrams.setdefault(dis, set()).add(prev_word)
                bigrams[dis].add(OOV_TERM)

        for word in word_sequence[1:]:
            bigrams.setdefault(prev_word, set()).add(word)

            if conservative:
                bigrams[prev_word].add(OOV_TERM)

            if disfluency:
                bigrams[prev_word].update(disfluencies)

                for dis in disfluencies:
                    bigrams[dis].add(word)

            prev_word = word

        # ...valid end
        bigrams.setdefault(prev_word, set()).add(OOV_TERM)

    node_ids = {}
    def get_node_id(word):
        node_id = node_ids.get(word, len(node_ids) + 1)
        node_ids[word] = node_id
        return node_id

    output = ""
    for from_word in sorted(bigrams.keys()):
        from_id = get_node_id(from_word)

        successors = bigrams[from_word]
        if len(successors) > 0:
            weight = -math.log(1.0 / len(successors))
        else:
            weight = 0

        for to_word in sorted(successors):
            to_id = get_node_id(to_word)
            output += '%d    %d    %s    %s    %f' % (from_id, to_id, to_word, to_word, weight)
            output += "\n"

    output += "%d    0\n" % (len(node_ids))

    return output.encode()

def make_transcript_fst(word_sequences, **kwargs):
    '''
    Use the given token sequence to make a bigram language model
    in OpenFST plain text format.

    When the "conservative" flag is set, an [oov] is interleaved
    between successive words.

    When the "disfluency" flag is set, a small set of disfluencies is
    interleaved between successive words

    `Word sequence` is a list of lists, each valid as a start
    '''

    if len(word_sequences) == 0 or type(word_sequences[0]) != list:
        word_sequences = [word_sequences]

    conservative = kwargs['conservative'] if 'conservative' in kwargs else False
    disfluency = kwargs['disfluency'] if 'disfluency' in kwargs else False
    disfluencies = kwargs['disfluencies'] if 'disfluencies' in kwargs else []

    bigrams = {OOV_TERM: set([OOV_TERM])}

    cur_node_id = 0
    next_node_id = 0
    end_node_id = sum([len(word_sequence) - 1 for word_sequence in word_sequences]) + 1
    next_available_node_id = 1
    output = []
    start_node_flag = True
    end_node_flag = True
    for word_sequence in word_sequences:
        if len(word_sequence) == 0:
            continue

        cur_node_id = 0  # start node
        if start_node_flag:
            if disfluency:
                for dis in disfluencies:
                    output.append((cur_node_id, cur_node_id, dis, dis, 0.0))
            
            if conservative:
                output.append((cur_node_id, cur_node_id, OOV_TERM, OOV_TERM, 0.0))

            start_node_flag = False
        
        if len(word_sequence) == 1:
            next_node_id = end_node_id
        else:
            next_node_id = next_available_node_id
            next_available_node_id += 1
        weight = 2.0
        for i, word in enumerate(word_sequence):
            output.append((cur_node_id, next_node_id, word, word, weight))
            output.append((cur_node_id, next_node_id, OOV_TERM, OOV_TERM, 0.0))
            # TODO: <eps>?

            if i == len(word_sequence) - 1:
                break
            elif i == len(word_sequence) - 2:
                cur_node_id = next_node_id
                next_node_id = end_node_id
            else:
                cur_node_id = next_node_id
                next_node_id = next_available_node_id
                next_available_node_id += 1

            if disfluency:
                for dis in disfluencies:
                    output.append((cur_node_id, cur_node_id, dis, dis, 0.0))
            
            if conservative:
                output.append((cur_node_id, cur_node_id, OOV_TERM, OOV_TERM, 0.0))
        
        if end_node_flag:
            if disfluency:
                for dis in disfluencies:
                    output.append((end_node_id, end_node_id, dis, dis, 0.0))
            
            if conservative:
                output.append((end_node_id, end_node_id, OOV_TERM, OOV_TERM, 0.0))

            end_node_flag = False

    assert end_node_id == next_available_node_id, f"{end_node_id} vs {next_available_node_id}"

    output.sort()

    output_str = "\n".join(map(lambda x: f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}", output))
    output_str += f"\n{end_node_id} 0\n"

    return output_str.encode()

def make_bigram_language_model(kaldi_seq, proto_langdir, **kwargs):
    """Generates a language model to fit the text.

    Returns the filename of the generated language model FST.
    The caller is resposible for removing the generated file.

    `proto_langdir` is a path to a directory containing prototype model data
    `kaldi_seq` is a list of words within kaldi's vocabulary.
    """

    # Generate a textual FST
    txt_fst = make_bigram_lm_fst(kaldi_seq, **kwargs)
    # txt_fst = make_transcript_fst(kaldi_seq, **kwargs)
    txt_fst_file = tempfile.NamedTemporaryFile(delete=False)
    txt_fst_file.write(txt_fst)
    txt_fst_file.close()
    # subprocess.check_output(["cp", txt_fst_file.name, "temp/1.fst"])

    hclg_filename = tempfile.mktemp(suffix='_HCLG.fst')
    # hclg_filename = "temp/1_HCLG.fst"
    try:
        devnull = open(os.devnull, 'wb')
        # print([MKGRAPH_PATH,
        #                 proto_langdir,
        #                 txt_fst_file.name,
        #                 hclg_filename])
        subprocess.check_output([MKGRAPH_PATH,
                        proto_langdir,
                        txt_fst_file.name,
                        hclg_filename],
                        stderr=devnull)
    except Exception as e:
        try:
            os.unlink(hclg_filename)
        except:
            pass
        raise e
    finally:
        os.unlink(txt_fst_file.name)

    return hclg_filename

if __name__=='__main__':
    import sys
    make_bigram_language_model(open(sys.argv[1]).read(), Resources().proto_langdir)
