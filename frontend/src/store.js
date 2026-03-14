import { create } from 'zustand';

const useAppStore = create((set) => {
    const makeSetter = (key) => (val) => set((state) => ({
        [key]: typeof val === 'function' ? val(state[key]) : val
    }));

    return {
        video1Url: "", setVideo1Url: makeSetter('video1Url'),
        video2Url: "", setVideo2Url: makeSetter('video2Url'),
        referenceSummary1: "", setReferenceSummary1: makeSetter('referenceSummary1'),
        referenceSummary2: "", setReferenceSummary2: makeSetter('referenceSummary2'),
        result1: null, setResult1: makeSetter('result1'),
        result2: null, setResult2: makeSetter('result2'),
        loading1: false, setLoading1: makeSetter('loading1'),
        loading2: false, setLoading2: makeSetter('loading2'),
        evaluating1: false, setEvaluating1: makeSetter('evaluating1'),
        evaluating2: false, setEvaluating2: makeSetter('evaluating2'),
        error1: null, setError1: makeSetter('error1'),
        error2: null, setError2: makeSetter('error2'),
        showSummary1: true, setShowSummary1: makeSetter('showSummary1'),
        showSummary2: true, setShowSummary2: makeSetter('showSummary2'),
        showGraph1: true, setShowGraph1: makeSetter('showGraph1'),
        showGraph2: true, setShowGraph2: makeSetter('showGraph2'),
        showCombinedText1: false, setShowCombinedText1: makeSetter('showCombinedText1'),
        showCombinedText2: false, setShowCombinedText2: makeSetter('showCombinedText2'),
        showEvaluation1: false, setShowEvaluation1: makeSetter('showEvaluation1'),
        showEvaluation2: false, setShowEvaluation2: makeSetter('showEvaluation2'),

        fusedReferenceSummary: "", setFusedReferenceSummary: makeSetter('fusedReferenceSummary'),
        showFusedEvaluation: false, setShowFusedEvaluation: makeSetter('showFusedEvaluation'),
        evaluatingFused: false, setEvaluatingFused: makeSetter('evaluatingFused'),
        fusedEvaluationResult: null, setFusedEvaluationResult: makeSetter('fusedEvaluationResult'),

        fusedResult: null, setFusedResult: makeSetter('fusedResult'),
        loadingFuse: false, setLoadingFuse: makeSetter('loadingFuse'),
        errorFuse: null, setErrorFuse: makeSetter('errorFuse'),
        showUnifiedSummary: true, setShowUnifiedSummary: makeSetter('showUnifiedSummary'),
        showUnifiedNotes: false, setShowUnifiedNotes: makeSetter('showUnifiedNotes'),
        showUnifiedGraph: true, setShowUnifiedGraph: makeSetter('showUnifiedGraph'),

        generatingFusedNotes: false, setGeneratingFusedNotes: makeSetter('generatingFusedNotes'),
        fusedNotesUrl: null, setFusedNotesUrl: makeSetter('fusedNotesUrl'),

        generatingFusedSummary: false, setGeneratingFusedSummary: makeSetter('generatingFusedSummary'),
        showFusedSummary: true, setShowFusedSummary: makeSetter('showFusedSummary'),

        nonKgSummary: "", setNonKgSummary: makeSetter('nonKgSummary'),
        generatingNonKgSummary: false, setGeneratingNonKgSummary: makeSetter('generatingNonKgSummary'),
        nonKgNotesUrl: null, setNonKgNotesUrl: makeSetter('nonKgNotesUrl'),
        generatingNonKgNotes: false, setGeneratingNonKgNotes: makeSetter('generatingNonKgNotes'),
        showNonKgSummary: false, setShowNonKgSummary: makeSetter('showNonKgSummary'),
        nonKgReferenceSummary: "", setNonKgReferenceSummary: makeSetter('nonKgReferenceSummary'),
        nonKgEvaluationResult: null, setNonKgEvaluationResult: makeSetter('nonKgEvaluationResult'),
        evaluatingNonKg: false, setEvaluatingNonKg: makeSetter('evaluatingNonKg'),
        showNonKgEvaluation: false, setShowNonKgEvaluation: makeSetter('showNonKgEvaluation'),

        kgNotesRefText: "", setKgNotesRefText: makeSetter('kgNotesRefText'),
        kgNotesEval: null, setKgNotesEval: makeSetter('kgNotesEval'),
        evaluatingKgNotes: false, setEvaluatingKgNotes: makeSetter('evaluatingKgNotes'),
        nonKgNotesRefText: "", setNonKgNotesRefText: makeSetter('nonKgNotesRefText'),
        nonKgNotesEval: null, setNonKgNotesEval: makeSetter('nonKgNotesEval'),
        evaluatingNonKgNotes: false, setEvaluatingNonKgNotes: makeSetter('evaluatingNonKgNotes'),

        statDataset: [], setStatDataset: makeSetter('statDataset'),
        statVideoName: "", setStatVideoName: makeSetter('statVideoName'),
        statKgRouge: "", setStatKgRouge: makeSetter('statKgRouge'),
        statNonKgRouge: "", setStatNonKgRouge: makeSetter('statNonKgRouge'),
        statHumanKg: "", setStatHumanKg: makeSetter('statHumanKg'),
        statHumanNonKg: "", setStatHumanNonKg: makeSetter('statHumanNonKg'),
        statSaving: false, setStatSaving: makeSetter('statSaving'),
        statRunning: false, setStatRunning: makeSetter('statRunning'),
        statVideos: [], setStatVideos: makeSetter('statVideos'),
        statResults: null, setStatResults: makeSetter('statResults'),
        statMessage: "", setStatMessage: makeSetter('statMessage'),
        datasetLoading: true, setDatasetLoading: makeSetter('datasetLoading'),
        datasetError: null, setDatasetError: makeSetter('datasetError'),
    };
});

export default useAppStore;
