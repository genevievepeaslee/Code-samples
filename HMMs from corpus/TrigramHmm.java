//patas

import java.io.*;
import java.util.*;

public class TrigramHmm {

	public static void main(String[] args) throws IOException {
		/**
		 * Infers a trigram HMM for POS tagging from tagged input text (System.in). 
		 * Emission and transition probabilities are MLEs with K-N smoothing.
		 * 
		 * Expected arguments: [output file] [lambda 1] [lambda 2] [lambda 3] [unk_prob_file]
		 * lambdas are interpolation weights, unk_prob_file contains probability of OOV word for each tag: P(<unk>|tag) 
		 */
		BufferedReader read_in = new BufferedReader(new InputStreamReader(System.in));		

		double lambda1 = Double.parseDouble(args[0]);
		double lambda2 = Double.parseDouble(args[1]);
		double lambda3 = Double.parseDouble(args[2]);
		BufferedReader unk_probs_file = new BufferedReader(new FileReader(new File(args[3])));

		// stores counts of each single tag, INCLUDING BOS and EOS
		HashMap<String, Integer> tag_counts = new HashMap<String, Integer>(); 
		
		//how many times each tag is followed by each other tag // T_(i-1) --> (T_i --> count)
		HashMap<String, HashMap<String, Integer>> t_tag_counts = new HashMap<String, HashMap<String, Integer>>(); 
		
		//how many times each two-tag sequence is followed by each other tag // T_(i-2) T_(i-1) --> (T_i --> count)
		HashMap<String, HashMap<String, Integer>> tt_tag_counts = new HashMap<String, HashMap<String, Integer>>(); 
		
		// key: tag_i | value: word_i
		HashMap<String, HashMap<String, Integer>> tag_word_counts = new HashMap<String, HashMap<String, Integer>>(); 
		
		//  key: tag; value: P(<unk>|tag)
		HashMap<String, Double> unk_probs = new HashMap<String, Double>(); 

		// read <unk> probs into HashMap
		String pair;
		while ((pair = unk_probs_file.readLine()) != null) {
			unk_probs.put(pair.substring(0, pair.indexOf(' ')), Double.parseDouble(pair.substring(pair.indexOf(' ') + 1)));
		}

		//keep track of words seen in training data (doesn't include BOS or EOS)
		HashSet<String> words = new HashSet<String>();

		String sentence;
		int all_tag_occurrences = 0; // sum of all word/tag tokens including BOS and EOS
		while ((sentence = read_in.readLine()) != null) {
			if (!sentence.equals("")) {
				sentence = sentence + " </s>/EOS </s>/EOS";
				String[] pairs = sentence.split("\\s+");
				all_tag_occurrences += pairs.length + 1; // includes EOS, BOS, and BOS for every sentence
				String t_1 = "BOS";
				String t_2 = "BOS";
				incrementTagTotal("BOS", tag_counts);
				incrementTagTotal("BOS", tag_counts);
				incrementCount("BOS", "<s>", tag_word_counts); //only count the second BOS as emitting
				incrementCount(t_2, t_1, t_tag_counts);
				for (int i = 0; i < pairs.length; i++) {
					// process word/tag pair
					String w = pairs[i].substring(0, pairs[i].lastIndexOf('/'));
					String t = pairs[i].substring(pairs[i].lastIndexOf('/') + 1);
					while (w.contains("\\")) {
						int slash = w.indexOf('\\');
						w = w.substring(0, slash) + w.substring(slash +1);
					}
					// Add w to set of all words and increment tag total of t; t_tag_count total of t following t_1;
					// tt_tag_count total of t following t_2 t_1; tag_word_count total of t -> w
					words.add(w);
					//everything except last EOS
					if (i != pairs.length - 1) {
						incrementTagTotal(t, tag_counts);
						incrementCount(t, w, tag_word_counts);
					}
					incrementCount(t_2 + " " + t_1, t, tt_tag_counts);
					incrementCount(t_1, t, t_tag_counts);
					//update t_2 and t_1
					t_2 = t_1;
					t_1 = t;
				}
			}
		}
		
		// generate transitions: (t u)	(u v)	0.xxxxx
		int num_tags = tag_counts.keySet().size() - 2; //subtract BOS and EOS
		Double P1 = 0.0;
		Double P2 = 0.0;
		Double P3 = 0.0;

		ArrayList<String> trans_outputs = new ArrayList<String>();
		for (String t_2 : tag_counts.keySet()) {
			for (String t_1 : tag_counts.keySet()) {
				Double sum = 0.0;
				for (String t : tag_counts.keySet()) {
					P1 = lambda1 * tag_counts.get(t) / all_tag_occurrences;
					P2 = 0.0; // default: if (t_1 t) is unseen
					if (t_tag_counts.containsKey(t_1)) { 
						if (t_tag_counts.get(t_1).containsKey(t)) { // if (t_1 t) is seen
							P2 = lambda2 * t_tag_counts.get(t_1).get(t) / tag_counts.get(t_1);
						}
					}
					P3 = 0.0; // default: (t_2 t_1) seen but never followed by t; (t_2 t_1) seen but followed by BOS
					if (tt_tag_counts.containsKey(t_2 + " " + t_1)) { // (t_2 t_1) bigram seen
						if (!t.equals("BOS") && tt_tag_counts.get(t_2 + " " + t_1).containsKey(t)) { // (t_2 t_1 t) trigram seen and t is not BOS
							P3 = lambda3 * tt_tag_counts.get(t_2 + " " + t_1).get(t) / t_tag_counts.get(t_2).get(t_1);
						} else { // seen (t_2 t_1) but not followed by t
							P3 = 0.0;
						}
					} else { // not seen (t_2 t_1)
						if (!t.equals("BOS")) {
							P3 = lambda3 * 1.0 / (num_tags + 1); // if (t_2 t_1) is unseen and t is not BOS [1 / (|T| + 1)]
						}
					}
					sum += (P1 + P2 + P3);
					trans_outputs.add(t_2 + "_" + t_1 + "\t" + t_1 + "_" + t + "\t" + (P1 + P2 + P3));
				}
			}
		}

		/*
		 * "If you don't see a tag in unk_prob_sec22, you just assume that P(<unk> | tag) = 0,
		 * and as a result, P_smooth(w | tag) = P(w | tag), where w is a known word.
		 * For a known word w, Psmooth(w | tag) = P(w | tag) * (1 − P(< unk >| tag)), where P(w | tag) = cnt(w,tag)/cnt(tag)."
		 */
		
		// generate emissions
		HashMap<String, ArrayList<String>> emission_probs = new HashMap<String, ArrayList<String>>(); 
		for (String t2 : tag_counts.keySet()) { 
			double p_unk = 0.0;
			if (unk_probs.containsKey(t2)) {
				p_unk = unk_probs.get(t2);
			}
			for (String w : tag_word_counts.get(t2).keySet()) {
				Double Psmooth = (double) tag_word_counts.get(t2).get(w) / tag_counts.get(t2) * (1 - p_unk); // smoothed P(w|t2)
				if (t2.equals("BOS")) {
					Psmooth = (double) tag_word_counts.get(t2).get(w) / (tag_counts.get(t2) / 2) * (1 - p_unk); // smoothed P(w|BOS) (have to divide tag_counts.get(BOS) by 2 because it gets added twice
				}
				for (String t1 : tag_counts.keySet()) { // t1_t2 emission prob for each w (same emission probability)
					add(t1 + "_" + t2, w + "\t" + Psmooth, emission_probs);
				}
			}
			if (p_unk > 0.0) {
				for (String t1 : tag_counts.keySet()) {
					add(t1 + "_" + t2, "<unk>\t" + p_unk, emission_probs); // t1_t2 emission prob for <unk>
				}
			}
		}
		
		//print header
		System.out.println("state_num=" + ((num_tags + 2) * (num_tags + 2))); // (any_tag + BOS + EOS) to (any_tag + BOS + EOS)
		System.out.println("sym_num=" + (words.size() + 2));
		System.out.println("init_line_num=1");
		System.out.println("trans_line_num=" + Math.pow((num_tags + 2), 3)); // (eachtag + BOS + EOS)_(eachtag + BOS + EOS)_(eachtag + BOS + EOS)
		//sum emission lines
		int num_emit_lines = 0;
		for (String t : emission_probs.keySet()) {
			for (String w : emission_probs.get(t)) {
				num_emit_lines++;
			}
		}
		
		System.out.println("emiss_line_num=" + num_emit_lines);

		// print initial
		System.out.println("\n\\init\nBOS_BOS\t1.0");

		//print transitions
		System.out.println("\n\n\n\\transition");
		for (String line : trans_outputs) {
			System.out.println(line);
		}
		
		//print emissions
		System.out.println("\n\\emission");
		for (String t1t2 : emission_probs.keySet()) {
			Double sum = 0.0;
			for (String w : emission_probs.get(t1t2)) {
				System.out.println(t1t2 + "\t" + w);
				sum += Double.parseDouble(w.substring(w.indexOf("\t")));
			}
			//System.out.println("sum " + sum);
		}
	}

	public static void add(String t1t2, String w_and_prob, HashMap<String, ArrayList<String>> emissions) {
		if (!emissions.containsKey(t1t2)) {
			emissions.put(t1t2, new ArrayList<String>());
		}
		emissions.get(t1t2).add(w_and_prob);
	}

	public static void incrementTagTotal(String tag, HashMap<String, Integer> tagCounts) {
		if (tagCounts.containsKey(tag)) {
			tagCounts.put(tag, tagCounts.get(tag) + 1);
		} else {
			tagCounts.put(tag, 1);
		}
	}

	public static void incrementCount(String key1, String key2, HashMap<String, HashMap<String, Integer>> map) {
		if (map.containsKey(key1)) {
			if (map.get(key1).containsKey(key2)) {
				map.get(key1).put(key2, map.get(key1).get(key2) + 1);
			} else {
				map.get(key1).put(key2, 1);
			}
		} else {
			map.put(key1, new HashMap<String, Integer>());
			map.get(key1).put(key2, 1);
		}
	}
}
