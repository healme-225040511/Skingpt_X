"""CPU-only protocol tests; no model access or network calls."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common

class ProtocolTests(unittest.TestCase):
    def test_parser(self):
        labels=['eczema','psoriasis','acne','rosacea','nevus']
        for response in ['{"PrimaryDiagnosis":"Eczema"}', '```json\n{"PrimaryDiagnosis":"eczema"}\n```',
                         'Result: {"PrimaryDiagnosis":"eczema (likely)"}']:
            self.assertEqual(common.parse_choice(response,labels),'eczema')
        for response in ['invalid','[]','{}','{"PrimaryDiagnosis":""}',
                         '{"PrimaryDiagnosis":"eczema or psoriasis"}']:
            self.assertIsNone(common.parse_choice(response,labels))

    def test_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'Synthetic';root.mkdir(); labels=['a','b','c','d','e']
            manifest=[]
            for i in range(6):
                content=f'image-{i}'.encode(); digest=hashlib.sha256(content).hexdigest()
                image=f'{i}.jpg';(root/image).write_bytes(content)
                manifest.append(dict(sample_id=digest,sha256=digest,image=image,
                                     split='train' if i<5 else 'test',label=labels[i%5]))
            m=manifest[-1]
            job=dict(job_id='test',sample_id=m['sample_id'],dataset='Synthetic',ground_truth=m['label'],
                     image=m['image'],image_sha256=m['sha256'],prompt='frozen evidence',
                     prompt_sha256=hashlib.sha256(b'frozen evidence').hexdigest(),
                     candidate_labels=labels,panderm_prediction='a')
            (root/'classes.json').write_text(json.dumps(labels))
            (root/'manifest.jsonl').write_text('\n'.join(map(json.dumps,manifest)))
            def write_job(j):
                (root/'case_jobs.jsonl').write_text(json.dumps(j))
                (root/'audit.json').write_text(json.dumps(dict(status='PASS',num_classes=5,
                    checksums={n:common.sha(root/n) for n in ['classes.json','manifest.jsonl','case_jobs.jsonl']})))
            write_job(job)
            with patch.dict(common.EXPECTED,{'Synthetic':(5,5,1)}):
                self.assertEqual(len(common.verify(root)),1)
                bad=dict(job,prompt='changed evidence');write_job(bad)
                with self.assertRaises(AssertionError):common.verify(root)
                write_job(dict(job,ground_truth='wrong'))
                with self.assertRaises(AssertionError):common.verify(root)
                write_job(job);(root/m['image']).write_bytes(b'tampered')
                with self.assertRaises(AssertionError):common.verify(root)

if __name__=='__main__':unittest.main()
