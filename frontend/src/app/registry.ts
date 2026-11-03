import {localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
export const groups = localized(()=>([tr('workbench.analysis_and_recording'), tr('workbench.synthesis_and_manipulation'), tr('workbench.annotation_and_experiments'), tr('workbench.literature_and_learning')] as const));
export interface Module {id:string;title:string;description:string;icon:string;group:0|1|2|3}
// Persisted workspace keys use these IDs. Display order must never assign identity.
export const modules:Module[] = localized(()=>([
  {id:'M16',title:tr('workbench.recording_157'),description:tr('workbench.local_capture_task_recordings_and_recoverable_editing'),icon:'microphone',group:0},
  {id:'M05',title:tr('workbench.lip_tracking'),description:tr('workbench.video_capture_tracking_and_alignment'),icon:'lip',group:0},
  {id:'M01',title:tr('workbench.parameter_estimation'),description:tr('workbench.batch_extraction_of_acoustic_and_lip_parameters'),icon:'sliders',group:0},
  {id:'M02',title:tr('workbench.parameter_display'),description:tr('workbench.compare_parameter_tracks_and_select_time_ranges'),icon:'chart',group:0},
  {id:'M03',title:tr('workbench.egg_signal_analysis'),description:tr('workbench.two_channel_signals_and_glottal_events'),icon:'wave',group:0},
  {id:'M04',title:tr('workbench.lpc_spectrum'),description:tr('workbench.linked_waveforms_and_formant_spectra'),icon:'bars',group:0},
  {id:'M06',title:tr('workbench.acoustic_synthesis'),description:tr('workbench.klatt_parametric_speech_synthesis'),icon:'speaker',group:1},
  {id:'M10',title:tr('workbench.articulatory_synthesis'),description:tr('workbench.vocal_tract_modeling_and_dynamic_articulation'),icon:'tract',group:1},
  {id:'M07',title:tr('workbench.phonation_synthesis'),description:tr('workbench.explore_continua_of_phonation_types'),icon:'glottis',group:1},
  {id:'M08',title:tr('workbench.time_and_pitch_manipulation'),description:tr('workbench.adjust_speaking_rate_and_f0_independently'),icon:'time-pitch',group:1},
  {id:'M09',title:tr('workbench.spectrogram_to_audio'),description:tr('workbench.reconstruct_audio_from_a_spectrogram'),icon:'spectrum-audio',group:1},
  {id:'M17',title:tr('workbench.ipa_chart_plus'),description:tr('workbench.complete_ipa_extipa_and_voqs_input_charts'),icon:'ipa-keyboard',group:2},
  {id:'M13',title:tr('workbench.chinese_to_ipa_transcription'),description:tr('workbench.convert_chinese_characters_to_ipa'),icon:'ipa',group:2},
  {id:'M11',title:tr('workbench.mfa_forced_alignment'),description:tr('workbench.model_based_forced_alignment'),icon:'forced-alignment',group:2},
  {id:'M12',title:tr('workbench.textgrid_annotation'),description:tr('workbench.textgrid_editing_and_lip_sync_alignment'),icon:'textgrid',group:2},
  {id:'M14',title:tr('workbench.phonological_analysis'),description:tr('workbench.character_tables_and_homophone_groupings'),icon:'nodes',group:2},
  {id:'M15',title:tr('workbench.perceptual_experiments'),description:tr('workbench.experiment_design_configuration_and_execution'),icon:'headphones',group:2},
  {id:'M18',title:tr('workbench.phonetics_paper_reader'),description:tr('workbench.retrieve_papers_by_date_and_read_originals_alongside_chinese'),icon:'book',group:3},
  {id:'M19',title:tr('workbench.language_maps'),description:tr('workbench.language_features_county_level_dialect_distributions_and_editable_research'),icon:'map',group:3},
]));
