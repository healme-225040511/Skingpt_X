"""Synthetic end-to-end CPU scorer regression; never accesses model weights."""
import contextlib,hashlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import score

class ScoringTests(unittest.TestCase):
    def test_paired_scoring_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);name='Synthetic';(root/name).mkdir()
            jobs=[]
            for i in range(4):
                jobs.append(dict(job_id=str(i),dataset=name,sample_id=str(i),ground_truth='a' if i<2 else 'b',
                    prompt_sha256='prompt',image_sha256='image',candidate_labels=['a','b','c','d','e'],
                    historical_8b_prediction=['b','a','a','b'][i],historical_8b_valid=True,panderm_prediction='a'))
            (root/name/'case_jobs.jsonl').write_text('\n'.join(map(json.dumps,jobs)))
            meta=dict(model_id='Qwen/Qwen3-VL-30B-A3B-Instruct-FP8',dataset=name,
                      jobs_sha256=score.sha(root/name/'case_jobs.jsonl'))
            sig=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
            output=root/'out.jsonl';output.with_suffix('.meta.json').write_text(json.dumps(meta))
            predictions=[]
            for i,j in enumerate(jobs):
                predictions.append(dict(job_id=j['job_id'],dataset=name,run_signature=sig,prompt_sha256='prompt',
                    image_sha256='image',response=json.dumps({'PrimaryDiagnosis':['a','b','b','b'][i]}),
                    seconds=1,input_tokens=5,output_tokens=3,finish_reason='stop'))
            output.write_text('\n'.join(map(json.dumps,predictions)))
            argv=['score.py','--data-root',str(root),'--outputs',str(output),'--report-dir',str(root/'report'),'--bootstrap','1000']
            with patch.object(score,'EXPECTED',{name:(5,5,4)}),patch.object(score,'verify',return_value=jobs),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                score.main()
            result=json.loads((root/'report/metrics.json').read_text())['datasets'][name]
            self.assertEqual(result['reference']['accuracy'],.5)
            self.assertEqual(result['target']['accuracy'],.75)
            self.assertEqual((result['rescue'],result['harm']),(2,1))
            self.assertEqual(result['accuracy_delta'],.25)
            output.write_text('\n'.join(map(json.dumps,predictions[:-1])))
            argv[argv.index('--report-dir')+1]=str(root/'incomplete')
            with patch.object(score,'EXPECTED',{name:(5,5,4)}),patch.object(score,'verify',return_value=jobs),patch.object(sys,'argv',argv):
                with self.assertRaises(AssertionError):score.main()

if __name__=='__main__':unittest.main()
