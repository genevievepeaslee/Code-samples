import java.io.*;
import java.util.*;

public class BigramHmm {

	public static void main(String[] args) throws IOException {
		/**
		 * Infers a bigram HMM for POS tagging from annotated input text (System.in).
		 * 
		 * Expected arguments: output file name 
		 * 
		 * Example input: (.txt file, one sentence per line)
		 * 	Pierre/NNP Vinken/NNP ,/, 61/CD years/NNS old/JJ ,/, will/MD join/VB the/DT board/NN as/IN a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD ./.
		 * 
		 * Output:
		 * 	state_num=nn
		 * 	sym_num=nn
		 * 	init_line_num=nn
		 * 	trans_line_num=nn
		 * 	emiss_line_num=nn
		 * 	## the number of states
		 * 	## the size of output symbol alphabet
		 * 	## the number of lines for the initial probability
		 * 	## the number of lines for the transition probability
		 * 	## the number of lines for the emission probability
		 * 
	     * 	\init
	     * 	state  prob  lg_prob  ## prob=\pi(state), lg_prob=lg(prob)
	     * 	...
	     * 	\transition
	     * 	from_state to_state prob lg_prob  ## prob=P(to_state | from_state)
	     * 	...
	     * 	\emission
	     * 	state symbol prob lg_prob         ## prob=P(symbol | state)
		 * 
		 * ----Probabilities come directly from frequencies: no smoothing---- 
		 */
		
		if (args.length == 0) { 
			System.err.println("no input file");
		} else {
			HashMap<String, Integer> tag_totals = new HashMap<String, Integer>();
			
			// for each tag, stores every other tag it precedes and the number of times such a sequence occurs
			// key: tag_i | value: tag_(i+1), count
			HashMap<String, HashMap<String, Integer>> tagtag_counts = new HashMap<String, HashMap<String, Integer>>(); 
			
			// for each tag, stores the words it occurs with and the number of such pairs 
			// key: tag_i | value: word_i, count
			HashMap<String, HashMap<String, Integer>> tagword_counts = new HashMap<String, HashMap<String, Integer>>(); 
			
			BufferedReader read_in = new BufferedReader(new InputStreamReader(System.in));		
			
			String line;
			HashSet<String> words = new HashSet<String>();
			while ((line = read_in.readLine()) != null) {
				if (!line.equals("")) {
					// first word/tag pair: increment totals for tag BOS; tag t_1; transition BOS -> t_1; and emission t_1 -> w_1
					String[] pairs = line.split("\\s+");
					
					String w1 = pairs[0].substring(0, pairs[0].lastIndexOf('/'));
					String t1 = pairs[0].substring(pairs[0].lastIndexOf('/') + 1);
					while (w1.contains("\\")) {
						int slash = w1.indexOf('\\');
						w1 = w1.substring(0, slash) + w1.substring(slash +1);
					}
					incrementTagTotal("BOS", tag_totals);
					incrementTagTotal(t1, tag_totals);
					incrementCount("BOS", t1, tagtag_counts);
					incrementCount(t1, w1, tagword_counts);
					words.add(w1);
					
					//rest of word/tag pairs
					String t_prev = t1;
					for (int i = 1; i < pairs.length; i++) {
						// isolate word and tag
						String w = pairs[i].substring(0, pairs[i].lastIndexOf('/'));
						String t = pairs[i].substring(pairs[i].lastIndexOf('/') + 1);
						while (w.contains("\\")) {
							int slash = w.indexOf('\\');
							w = w.substring(0, slash) + w.substring(slash +1);
						}
						// add word to set of all words and increment totals of tag t, transition t_prev -> t, and emission t -> w
						words.add(w);
						incrementTagTotal(t, tag_totals);
						incrementCount(t_prev, t, tagtag_counts);
						incrementCount(t, w, tagword_counts);
						//update t_prev
						t_prev = t;
					}
					// increment totals of tag EOS and transition t_prev -> EOS
					incrementTagTotal("EOS", tag_totals);
					incrementCount(t_prev, "EOS", tagtag_counts);
				}
			}
			
			//sum number of transitions
			int trans_lines = 0;
			for (String tag1 : tagtag_counts.keySet()) {
				trans_lines += tagtag_counts.get(tag1).size();
			}
				
			//sum number of emissions
			int emit_lines = 0;
			for (String tag : tagword_counts.keySet()) {
				emit_lines += tagword_counts.get(tag).size();
			}
			
			//print header
			System.out.println("state_num=" + tag_totals.keySet().size());
			System.out.println("sym_num=" + words.size());
			System.out.println("init_line_num=" + 1);
			System.out.println("trans_line_num=" + trans_lines);
			System.out.println("emiss_line_num=" + emit_lines);
			
			// print initial state
			System.out.println("\n\\init\nBOS\t1.0");
			
			//print transitions
			System.out.println("\n\n\n\\transition");
			for (String t1 : tagtag_counts.keySet()) {
				for (String t2 : tagtag_counts.get(t1).keySet()) {
					System.out.println(t1 + "\t" + t2 + "\t" + (double) tagtag_counts.get(t1).get(t2) / tag_totals.get(t1));
				}
			}
			
			//print emissions
			System.out.println("\n\\emission");
			for (String t : tagword_counts.keySet()) {
				for (String w : tagword_counts.get(t).keySet()) {
					System.out.println(t + "\t" + w + "\t" + (double) tagword_counts.get(t).get(w) / tag_totals.get(t));
				}
			}	
		}
	
	public static void incrementTagTotal(String tag, HashMap<String, Integer> tagCounts) {
		if (tagCounts.containsKey(tag)) {
			tagCounts.put(tag, tagCounts.get(tag) + 1);
		} else {
			tagCounts.put(tag, 1);
		}
	}
	
	
	public static void incrementCount(String tag, String tag2_word, HashMap<String, HashMap<String, Integer>> map) {
		/**
		 * increments the count at map[tag][tag2_word]
		 * 
		 * @param tag2_word either tag)(i_1) OR word associated with <tag>
		 */
		if (map.containsKey(tag)) {
			if (map.get(tag).containsKey(tag2_word)) {
				map.get(tag).put(tag2_word, map.get(tag).get(tag2_word) + 1);
			} else {
				map.get(tag).put(tag2_word, 1);
			}
		} else {
			map.put(tag, new HashMap<String, Integer>());
			map.get(tag).put(tag2_word, 1);
		}
	}

}
