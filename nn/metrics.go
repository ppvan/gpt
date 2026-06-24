package nn

import "fmt"

func accuracy(pred, y Mat) float64 {
	correct := 0
	for i := range pred.Rows {
		if int(pred.Get(i, 0)) == int(y.Get(i, 0)) {
			correct++
		}
	}
	return float64(correct) / float64(pred.Rows)
}

func precision(pred, y Mat) float64 {
	n := pred.Rows
	classes := map[int]float64{}
	for i := range y.Rows {
		classes[int(y.Get(i, 0))] = 0
	}
	for class := range classes {
		truePositives := 0
		positives := 0
		for i := range n {
			prediction := int(pred.Get(i, 0))
			label := int(y.Get(i, 0))
			if prediction == class {
				positives++
			}
			if prediction == class && label == class {
				truePositives++
			}
		}
		if positives == 0 {
			classes[class] = 0
			continue
		}
		classes[class] = float64(truePositives) / float64(positives)
	}
	avg := 0.0
	for _, v := range classes {
		avg += v
	}
	return avg / float64(len(classes))
}

func recall(pred, y Mat) float64 {
	n := pred.Rows
	classes := map[int]float64{}
	for i := range y.Rows {
		classes[int(y.Get(i, 0))] = 0
	}
	for class := range classes {
		truePositives := 0
		actuals := 0
		for i := range n {
			prediction := int(pred.Get(i, 0))
			label := int(y.Get(i, 0))
			if label == class {
				actuals++
			}
			if prediction == class && label == class {
				truePositives++
			}
		}
		if actuals == 0 {
			classes[class] = 0
			continue
		}
		classes[class] = float64(truePositives) / float64(actuals)
	}
	avg := 0.0
	for _, v := range classes {
		avg += v
	}
	return avg / float64(len(classes))
}

func f1(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

func (e EvalMetrics) String() string {
	return fmt.Sprintf(
		"Metrics {\n"+
			"\tacc:  %.2f%%\n"+
			"\tprec: %.2f%%\n"+
			"\trec:  %.2f%%\n"+
			"\tf1:   %.2f%%\n"+
			"}",
		e.Accuracy*100,
		e.Precision*100,
		e.Recall*100,
		e.F1*100,
	)
}
