import java.util.*;
import java.io.*;

public class ViterbiPos {
	/**
	 * Uses the Viterbi algorithm to tag parts of speech on input text with an HMM provided in a text format
	 */

	public static void main(String[] args) throws IOException {
		// create HMM object from file
		File f = new File(args[0]);
		Hmm model = new Hmm(f);
		// run Viterbi on input text and print most likely sequence of tags
		BufferedReader test_file = new BufferedReader(new FileReader(new File(args[1])));
		String sentence;
		while ((sentence = test_file.readLine()) != null) {
			String[] observation = sentence.split("\\s+");
			System.out.print(sentence + " => ");
			model.viterbi(observation);
		}
	}

	public static class Hmm {
		// states represent pairs of tags (ex. VB_NN)
		
		// keys: all states that can be initial states (non-emitting) 
		// values: probability of state being initial state
		HashMap<String, Double> initials; 
		
		// keys: all to_states in HMM
		// values: states that can precede given state and associated transition probability  
		// transitions.get(x).keySet = every state than can precede x
		HashMap<String, HashMap<String, Double>> transitions;
		
		// keys: all words that HMM emits
		// values: states that can emit given word and associated emission probability 
		//emissions.get(w).keySet = every state that can emit w
		HashMap<String, HashMap<String, Double>> emissions; 
		
		// ALL states in hmm
		HashSet<String> states; 

		public Hmm (File f) throws IOException {
			/**
			 * constructs HMM from text file
			 * 
			 * ex.
			 * \init
			 * BOS_BOS 1.0
			 * 			
			 * \transition
			 * BOS_BOS	BOS_NN	0.121905632990675	-0.913976226126943	## p2=57/1921=0.0296720458094742 p1=6131/50293=0.121905632990675
			 * BOS_BOS	BOS_NNP	0.101485296164476	-0.993596876541071	## p2=375/1921=0.195210827693909 p1=5104/50293=0.101485296164476
			 * 
			 * \emission
			 * JJS_VB	announce	0.00136445205160047	-2.8650417213288	## 2/1178=0.00169779286926995 unk=0.196337741607325
			 * PRP_IN	are	0.000205230306388524	-3.68775850649665	## 1/4852=0.000206100577081616 unk=0.00422255340288127
			 */
			
			initials = new HashMap<String, Double>();
			transitions = new HashMap<String, HashMap<String, Double>>();
			emissions = new HashMap<String, HashMap<String, Double>>();
			states = new HashSet<String>();

			BufferedReader hmm_file = new BufferedReader(new FileReader(f));
			String line;
			// STORE INITIAL PROBS
			// skip comment lines at beginning of file
			line = hmm_file.readLine();
			while (!line.trim().equals("\\init")) {
				line = hmm_file.readLine();
			}
			line = hmm_file.readLine(); // first initial line
			while (!line.trim().equals("\\transition")) { // remaining initial lines
				if (!line.trim().equals("")) {
					String[] split = line.split("\\s+");
					// add to set of all states
					if (!states.contains(split[0])) {
						states.add(split[0]);
					}
					initials.put(split[0], Double.parseDouble(split[1]));
				}
				line = hmm_file.readLine();
			}

			// at this point, line = "/transition"
			line = hmm_file.readLine(); // first transition line

			// STORE TRANSITIONS
			while (!line.trim().equals("\\emission")) {
				if (!line.trim().equals("")) {
					String[] split = line.split("\\s+");
					Double prob = Double.parseDouble(split[2]);

					if (prob < 0.0 || prob > 1.0) {
						System.out.println("warning: the prob is not in [0,1] range: " + line);
					}
					// store states
					if (!states.contains(split[0])) {
						states.add(split[0]);
					}
					if (!states.contains(split[1])) {
						states.add(split[1]);
					}
					// store transition prob in matrix
					if (!transitions.containsKey(split[1])) {
						transitions.put(split[1], new HashMap<String, Double>());
					}
					transitions.get(split[1]).put(split[0], prob);
				}
				line = hmm_file.readLine();
			}

			// here line = "/emission"
			line = hmm_file.readLine(); // first emission line

			// STORE EMISSIONS
			while (line != null) { // all emission lines
				if (!line.trim().equals("")) { // except blank ones 
					String[] split = line.split("\\s+");
					Double prob = Double.parseDouble(split[2]);

					if (prob < 0.0 || prob > 1.0) {
						System.out.println("warning: the prob is not in [0,1] range: " + line);
					}
					if (!states.contains(split[0])) {
						states.add(split[0]);
					}
					// store emission prob in matrix
					if (!emissions.containsKey(split[1])) {
						emissions.put(split[1], new HashMap<String, Double>());
					}
					emissions.get(split[1]).put(split[0], prob);
				}
				line = hmm_file.readLine();
			}
		}


		public void viterbi(String[] observation) {
			/**
			 * Viterbi decoder. Prints most likely sequence of states that would produce sequence of words in <observation>. 
			 */
			// keys: ALL states in HMM
			// values: double[i] = best_prob(being in this state at observation[i]) (LOG probs)
			HashMap<String, double[]> trellis = new HashMap<String, double[]>();
			
			// keys: all hmm states (could make it all hmm to_states - better?)
			// values = most likely previous state, considering observation[index]
			HashMap<String, String[]> backtrace = new HashMap<String, String[]>(); 

			// initialize trellis backtrace
			for (String s : states) { 
				backtrace.put(s, new String[observation.length]);
				trellis.put(s, new double[observation.length]);
			}

			// process first word
			String word = observation[0];
			if (!emissions.containsKey(observation[0])) {
				word = "<unk>";
			}

			for (String emit_state : emissions.get(word).keySet()) { // consider each state that can emit first word
				double max_prob = -10000.0;
				String best_prev_state = "";
				double emit_prob = Math.log10(emissions.get(word).get(emit_state));
				for (String prev : transitions.get(emit_state).keySet()) { // consider each state that can precede emit_state
					if (initials.containsKey(prev)) { // if prev is an initial state, calculate overall prob and keep max
						double init_prob = Math.log10(initials.get(prev)); 
						double trans_prob = Math.log10(transitions.get(emit_state).get(prev));
						double overall_prob = emit_prob + trans_prob + init_prob;
						if (overall_prob > max_prob) { 
							max_prob = overall_prob;
							best_prev_state = prev;
						}
					}
				}
				// store most likely previous state and its probability
				if (!best_prev_state.equals("")) {
					trellis.get(emit_state)[0] = max_prob;
					backtrace.get(emit_state)[0] = best_prev_state;
				}
			}

			// repeat for remaining observations
			for (int i = 1; i < observation.length; i++) {
				word = observation[i];
				if (!emissions.containsKey(observation[i])) {
					word = "<unk>";
				}
				for (String emit_state : emissions.get(word).keySet()) { // consider each state that can emit this word
					double max_prob = -10000.0;
					String best_prev_state = "";
					double emit_prob = Math.log10(emissions.get(word).get(emit_state));
					for (String prev : transitions.get(emit_state).keySet()) { // consider each state that can precede emit_state
						if (trellis.get(prev)[i-1] != 0.0) { // if it's possible to be in state prev on previous word...
							double trans_prob = Math.log10(transitions.get(emit_state).get(prev));
							double prev_prob = trellis.get(prev)[i-1]; // already a logprob
							double overall_prob = emit_prob + trans_prob + prev_prob;
							if (overall_prob > max_prob && overall_prob != 0.0) {
								max_prob = overall_prob;
								best_prev_state = prev;
							}
						}
					}
					// if haven't found a way to get to this emit state, trellis and backtrace entries are left empty
					if (!best_prev_state.equals("")) {  
						trellis.get(emit_state)[i] = max_prob;
						backtrace.get(emit_state)[i] = best_prev_state;
					}
				}
			}

			// trellis is completed; choose state with highest probability at final time step 
			double max_final_prob = -10000.0;
			String best_final_state = "";
			for (String state : trellis.keySet()) {
				if (trellis.get(state)[observation.length - 1] > max_final_prob && trellis.get(state)[observation.length - 1] != 0.0) {
					max_final_prob = trellis.get(state)[observation.length - 1];
					best_final_state = state;
				}
			}

			// FOR DEBUGGING
			//print backtrace
			//			System.out.print("\n\t\t");
			//			for (int i = 0; i < observation.length; i++) {
			//				System.out.print(i + "\t\t");
			//			}
			//			System.out.println();
			//			for (String s : backtrace.keySet()) {
			//				System.out.print(s + "\t\t");
			//				for (int j = 0; j < observation.length; j++) {
			//					System.out.print(backtrace.get(s)[j] + "\t\t");
			//				}
			//				System.out.println();
			//			}

			//print trellis
			//			System.out.println();
			//			for (String state : trellis.keySet()) {
			//				System.out.print(state + "\t\t");
			//				for (int i = 0; i < observation.length; i++) {
			//					System.out.print(trellis.get(state)[i] + "[" + i + "]\t");
			//				}
			//				System.out.println();
			//			}

			//print results
			String sequence = best_final_state;
			String current = best_final_state;
			for (int i = observation.length - 1; i >= 0; i--) {				
				String prev = backtrace.get(current)[i];
				sequence = prev + " " + sequence;
				current = prev;
			}
			System.out.println(sequence + " " + max_final_prob);
		}
	}
}

