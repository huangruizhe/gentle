import argparse
import logging
import os
import sys
from tqdm import tqdm

import gentle
import lhotse

parser = argparse.ArgumentParser(
        description='Align a transcript to audio by generating a new language model.  Outputs JSON')
parser.add_argument(
        '-o', '--output', metavar='output', type=str, 
        help='output filename')
parser.add_argument(
        '--conservative', dest='conservative', action='store_true',
        help='conservative alignment')
parser.set_defaults(conservative=False)
parser.add_argument(
        '--disfluency', dest='disfluency', action='store_true',
        help='include disfluencies (uh, um) in alignment')
parser.set_defaults(disfluency=False)
parser.add_argument(
        '--log', default="INFO",
        help='the log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)')
parser.add_argument(
        '--jobs', default="1:1",
        help='job_id:total_jobs')
parser.add_argument(
        '--margin', default="0", type=float,
        help='the duration of margin to pad before and after each cut')
parser.add_argument(
        'cuts', type=str,
        help='The lhotse cuts')
args = parser.parse_args()

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)
log_level = args.log.upper()
logging.getLogger().setLevel(log_level)

logging.info(args)

disfluencies = set(['uh', 'um'])

def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


cuts = lhotse.load_manifest(args.cuts)
cuts.describe()

resources = gentle.Resources()
logging.info("converting audio to 8K sampled wav")

job_id, total_jobs = list(map(int, args.jobs.split(":")))

total_processed = 0
for icut, cut in enumerate(cuts):
    if (icut % total_jobs) + 1 != job_id:
        continue
    #if (int(icut/10) % total_jobs) + 1 != job_id:
    #    continue
    #if icut < 28950:
    #    continue
    
    recording_id = cut.recording.id
    uid = cut.supervisions[0].id
    if os.path.exists(f"{args.output}/{recording_id}/{uid}.json"):
        continue

    assert len(cut.supervisions) == 1
    transcript = cut.supervisions[0].text
    audiofile = cut.recording.sources[0].source
    offset = max(0, cut.start + cut.supervisions[0].start - args.margin)
    duration = cut.supervisions[0].duration + 2 * args.margin
    
    with gentle.resampled(audiofile, offset=offset, duration=duration) as wavfile:
        logging.info(f" ========== {icut}/{len(cuts)}:{total_processed} {cut.supervisions[0].id} offset={offset} dur={duration} ==========")
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, disfluency=args.disfluency, conservative=args.conservative, disfluencies=disfluencies)
        result = aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)

    output_dir = f"{args.output}/{recording_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output = f"{output_dir}/{uid}.json"
    fh = open(output, 'w', encoding="utf-8") if args.output else sys.stdout
    fh.write(result.to_json(indent=2))

    total_processed += 1
#     if args.output:
#         logging.info("output written to %s" % (output))

logging.info(f"Done: {total_processed} processed")
