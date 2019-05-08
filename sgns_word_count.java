import java.io.*;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern; 

class sgns_word_count {
	private static String trainingDataFilePath = "F:\\word2vec\\rel\\unionfile";
	private static String modelFilePath = "F:\\word2vec\\rel\\vecIG";
	private static String outputFilePath = "F:\\word2vec\\rel\\outputFile";
	public static long word_num;
	public static long pair_num;
	public static long Outputnum;
	public static long window_size = 5;
	public static long min_reduce = 10;
	public static HashMap<String, Long> w_ucount = new HashMap<String, Long>();
	public static HashMap<String, Long> wordcount = new HashMap<String, Long>();
	public static HashMap<String, Double> wordweight = new HashMap<String, Double>();
	public static HashMap<String, Integer> tempword = new HashMap<String, Integer>();
	public static HashMap<Integer, Long> doclines = new HashMap<Integer, Long>();
	public static String Method;
	
	public static void main(String[] args) throws Exception {
    	long startTime = System.currentTimeMillis(); 
    	word_num = 0;
    	Outputnum = 0;
    	Method = "IG";
    	//Readpair();
    	LearnVocabFromCorpus();
    	LearnFromCorpus();
        long endTime = System.currentTimeMillis(); 
        System.out.println("程序运行时间：" + (endTime - startTime) + "ms");
    }
	
	public static void Readpairnum() {
		long cc=0;
		for(int i=0;i<=Outputnum;i++) {
			System.out.println("Reading Outputdoc"+i +" Start\n");
			cc=0;
			try {
				FileInputStream inputstream = new FileInputStream(outputFilePath+i);
				BufferedReader bufferreader = new BufferedReader(new InputStreamReader(inputstream,"UTF-8"));
				String line;
				line = bufferreader.readLine();
				while(line!=null) {
					cc++;
					String [] words = line.split(" |\t");
					pair_num+=Long.parseLong(words[3]);
					line = bufferreader.readLine(); 
				}
				inputstream.close();
			}catch (IOException e) {
				e.printStackTrace();
			}
			System.out.println("Reading Outputdoc"+i + " Finished!\n");
			doclines.put(i, cc);
		}
	}
	
	public static void FILESREAD_CAL() {
		long cc=0;
		for(int i=0;i<=Outputnum;i++) {
			System.out.println("Reading Outputdoc"+i +" Start\n");
			try {
				FileInputStream inputstream = new FileInputStream(outputFilePath+i);
				BufferedReader bufferreader = new BufferedReader(new InputStreamReader(inputstream,"UTF-8"));
				String line;
				line = bufferreader.readLine();
				while(line!=null) {
					cc++;
					if(cc % 1000000 == 0) {
						System.out.println("\tHas Read " + cc * 1.0 / doclines.get(i) + "%\n");
					}
					String [] words = line.split(" |\t");
					if(tempword.get(words[0]) != null) {
						String tem = words[0] + " " + words[1];
						long value = Long.parseLong(words[3]);
						if(w_ucount.get(tem) == null) w_ucount.put(tem, value);
						else w_ucount.put(tem, w_ucount.get(tem) + 1L);
					}
					line = bufferreader.readLine(); 
				}
				inputstream.close();
			}catch (IOException e) {
				e.printStackTrace();
			}
			System.out.println("Reading Outputdoc"+i + " Finished!\n");
		}
		calweight();
	}
	
	public static void Readpair() {
		Outputnum = 5;
		w_ucount.clear();
		Readpairnum();
		long cnt = 0;
		Iterator<String> keyIter = wordcount.keySet().iterator();
		while (keyIter.hasNext()) {
			String key = keyIter.next();
			tempword.put(key, 1);
			cnt += wordcount.get(key);
			if(cnt * 1.0 / word_num > 1.0 / (Outputnum * 2)) {
				FILESREAD_CAL();
				cnt=0;
				tempword.clear();
				w_ucount.clear();
			}
		}
		if(cnt!=0) {
			FILESREAD_CAL();
			tempword.clear();
			w_ucount.clear();
		}
	}
	
	public static void calweight()
	{
		Iterator<String> keyIter = w_ucount.keySet().iterator();
		while (keyIter.hasNext()) {
            String key = keyIter.next();
            String[] tem = key.split(" ");
        	if(wordcount.get(tem[0]) < min_reduce || wordcount.get(tem[1]) < min_reduce) continue;
            
        	if(wordweight.get(tem[0]) == null) {
        		wordweight.put(tem[0], 0.0);
        	}
        	//IG
        	if(Method == "IG") {
        		double pw = wordcount.get(tem[0]) * 1.0 / word_num;
        		double pu = wordcount.get(tem[1]) * 1.0 / word_num;
        		double p_w = 1 - pw;
        		double puw = (w_ucount.get(key)*1.0 / pair_num) / pw;
        		double pu_w = (wordcount.get(tem[1]) - w_ucount.get(key) * 1.0 / pair_num) / p_w;
        		double IG = -pu * Math.log(pu) + pw * puw * Math.log(puw) + p_w * pu_w *Math.log(pu_w);
        		wordweight.put(tem[0], wordweight.get(tem[0])+IG);
        	}
        	//MI
        	if(Method == "MI") {
        		long A = w_ucount.get(key);
        		long B = wordcount.get(tem[0]) - A;
        		long C = wordcount.get(tem[1]) - A;
        		long N = pair_num;
        		double MI = Math.log(A * N * 1.0 / (A + C) * (A + B));
        		wordweight.put(tem[0], wordweight.get(tem[0])+MI);
        	}
        	//CHI
        	if(Method == "CHI") {
        		long A = w_ucount.get(key);
        		long B = wordcount.get(tem[0]) - A;
        		long C = wordcount.get(tem[1]) - A;
        		long N = pair_num;
        		long D = N - wordcount.get(tem[0]) - wordcount.get(tem[1]) + A;
        		double CHI = (N * (A * D - C * B) * (A * D - C * B) * 1.0) / ((A + C) * (B + D) * (A + B) * (C + D));
        		wordweight.put(tem[0], wordweight.get(tem[0])+CHI);
        	}
        }
	}
	public static void outputweight()
	{
		Iterator<String> OutIter = wordweight.keySet().iterator();
		BufferedWriter bw = null;
		
		try {
        	FileOutputStream writerStream = new FileOutputStream(modelFilePath);    
            bw = new BufferedWriter(new OutputStreamWriter(writerStream, "UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        while(OutIter.hasNext()) {
        	String key = OutIter.next();
        	double value = wordweight.get(key);
        	long count = wordcount.get(key);
        	if(count < min_reduce) continue;
        	try {
        		bw.append(key + "\t" + count +"\t" + value + "\r\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            bw.flush();
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	
	public static void OutputPair() {
		Iterator<String> OutIter = w_ucount.keySet().iterator();
		BufferedWriter bw = null;
		String tem = outputFilePath + Outputnum;
		System.out.println(tem);
		Outputnum += 1;
		try {
        	FileOutputStream writerStream = new FileOutputStream(tem);    
            bw = new BufferedWriter(new OutputStreamWriter(writerStream, "UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        while(OutIter.hasNext()) {
        	String key = OutIter.next();
        	long value = w_ucount.get(key);
        	try {
        		bw.append(key + "\t" + value + "\r\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            bw.flush();
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        w_ucount.clear();
	}
	
	public static void LearnVocabFromCorpus() {
		try {
			FileInputStream inputstream = new FileInputStream(trainingDataFilePath);
			BufferedReader bufferreader = new BufferedReader(new InputStreamReader(inputstream,"UTF-8"));
			String line;
			line = bufferreader.readLine();
			while(line!=null) {
				String [] words = line.split("\\ |\t");
				for(int i=0;i<words.length;i++) {
					if(word_num % 1000000 == 0) {
						System.out.print("\tword_num : " + word_num/1000000 + "M " +"\n");
					}
					if(wordcount.get(words[i]) == null) wordcount.put(words[i],1L);
					else wordcount.put(words[i],wordcount.get(words[i])+1L);
					word_num++;
				}
				line = bufferreader.readLine();
			}
			inputstream.close();
		}catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void LearnFromCorpus()
	{
		String [] circur = new String[11];
		int num=0;
		long startTime2 = System.currentTimeMillis(); 
		try {
			FileInputStream inputstream = new FileInputStream(trainingDataFilePath);
			BufferedReader bufferreader = new BufferedReader(new InputStreamReader(inputstream,"UTF-8"));
			String line;
			line = bufferreader.readLine();
			while(line!=null) {
				String [] words = line.split("\\ |\t");
				for(int i=0;i<words.length;i++) {
					if(wordcount.get(words[i]) < min_reduce) continue;
					if(num % 1000000 == 0) {
						long endTime2 = System.currentTimeMillis();
						long TIME = endTime2 - startTime2;
						if(TIME > 600000) {
							System.out.println("output");
							OutputPair();
						}
						System.out.print("\tword_num : " + num/100000 + "M " + "pair_num : " + w_ucount.size() +"\n");
						startTime2 = System.currentTimeMillis(); 
					}
					if(words.length - i <= window_size) {
						circur[(int)window_size + i - words.length] = words[i];
					}
					for(int j=1;j<=window_size;j++) {
						int uid = i-j;
						if(uid < 0) {
							if(num<window_size) {
								break;
							}
							else {
								uid = uid + (int)window_size;
								if(wordcount.get(circur[uid]) == null || wordcount.get(circur[uid]) <min_reduce) continue;
								String tem = words[i]+" "+circur[uid];
								if(w_ucount.get(tem) == null) w_ucount.put(tem,1L);
								else w_ucount.put(tem,w_ucount.get(tem)+1);
								
								tem = circur[uid]+" "+words[i];
								if(w_ucount.get(tem) == null) w_ucount.put(tem,1L);
								else w_ucount.put(tem,w_ucount.get(tem)+1);
								pair_num += 2;
								continue;
							}
						}
						if(wordcount.get(words[uid]) < min_reduce) continue;
						String tem = words[i]+" "+words[uid];
						if(w_ucount.get(tem) == null) w_ucount.put(tem,1L);
						else w_ucount.put(tem,w_ucount.get(tem)+1);
						
						tem = words[uid]+" "+words[i];
						if(w_ucount.get(tem) == null) w_ucount.put(tem,1L);
						else w_ucount.put(tem,w_ucount.get(tem)+1);
						pair_num += 2;
					}
					num++;
				}
				line = bufferreader.readLine();
			}
			inputstream.close();
		}catch (IOException e) {
			e.printStackTrace();
		}
		if(Outputnum != 0){
			System.out.println("output");
			OutputPair();
		}
		if(Outputnum == 0){
			calweight();
			outputweight();
		}
	}
}
