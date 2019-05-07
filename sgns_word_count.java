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
	public static HashMap<String, Long> w_ucount = new HashMap<String, Long>();
	public static HashMap<String, Long> wordcount = new HashMap<String, Long>();
	public static HashMap<String, Double> wordweight = new HashMap<String, Double>();
	public static String Method;
	
	public static void main(String[] args) throws Exception {
    	long startTime = System.currentTimeMillis(); 
    	Pattern pattern = Pattern.compile("[0-9]*");
    	word_num = 0;
    	Outputnum = 0;
    	Method = "IG";
    	for(int i=0;i<args.length;i++) {
    		if(args[i] == "-window") {
    			Matcher isNum = pattern.matcher(args[i+1]);
    			if( !isNum.matches() ){
    				System.out.println("loss parameters!\n");
    				return;
    	        }
    			window_size = Integer.parseInt(args[i+1]);
    		}
    		if(args[i] == "-method") {
    			if(args[i+1] == "IG" 
    					|| args[i+1] == "MI"
    					|| args[i+1] == "CHI") {
    				Method = args[i+1];
    			}else {
    				System.out.println("False parameters!\n");
    				return;
    			}
    		}
    	}
    	LearnFromCorpus();
        long endTime = System.currentTimeMillis(); 
        System.out.println("程序运行时间：" + (endTime - startTime) + "ms");
    }
	public static void calweight()
	{
		Iterator<String> keyIter = w_ucount.keySet().iterator();
		while (keyIter.hasNext()) {
            String key = keyIter.next();
            String[] tem = key.split(" ");
        	if(wordcount.get(tem[0]) < 5 || wordcount.get(tem[1]) < 5) continue;
            
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
	
	public static void LearnFromCorpus()
	{
		String [] circur = new String[11];
		long startTime2 = System.currentTimeMillis(); 
		try {
			FileInputStream inputstream = new FileInputStream(trainingDataFilePath);
			BufferedReader bufferreader = new BufferedReader(new InputStreamReader(inputstream,"UTF-8"));
			String line;
			line = bufferreader.readLine();
			while(line!=null) {
				String [] words = line.split("\\ |\t");
				for(int i=0;i<words.length;i++) {
					if(word_num % 100000 == 0) {
						long endTime2 = System.currentTimeMillis();
						long TIME = endTime2 - startTime2;
						if(TIME > 60000) {
							System.out.println("output");
							OutputPair();
						}
						System.out.print("\tword_num : " + word_num/1000 + "K " + "pair_num : " + w_ucount.size() +"\n");
						startTime2 = System.currentTimeMillis(); 
					}
					if(wordcount.get(words[i]) == null) wordcount.put(words[i],1L);
					else wordcount.put(words[i],wordcount.get(words[i])+1);
					
					if(words.length - i <= window_size) {
						circur[(int)window_size + i - words.length] = words[i];
					}
					for(int j=1;j<=window_size;j++) {
						int uid = i-j;
						if(uid < 0) {
							if(word_num<window_size) {
								break;
							}
							else {
								uid = uid + (int)window_size;
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
						String tem = words[i]+" "+words[uid];
						if(w_ucount.get(tem) == null) w_ucount.put(tem,1L);
						else w_ucount.put(tem,w_ucount.get(tem)+1);
						
						tem = words[uid]+" "+words[i];
						if(w_ucount.get(tem) == null) w_ucount.put(tem,1L);
						else w_ucount.put(tem,w_ucount.get(tem)+1);
						pair_num += 2;
					}
					word_num++;
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
