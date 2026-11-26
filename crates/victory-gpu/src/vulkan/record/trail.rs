//! Diagnostics: what each command buffer records, kept with it (the last
//! `Trail::MOST` notes) until it is submitted; a submission that does not
//! finish is reported with its notes (`engine.rs`, `report_stalls`).

use super::Rec;

impl Rec {
    /// Notes `what` in the command buffer's trail, with the pipeline's
    /// shader names when they change.
    pub(super) fn note(&mut self, what: impl FnOnce() -> String) {
        self.note_as(what, false);
    }

    /// [`Rec::note`], with the shader names always (`named`): a dispatch,
    /// so that each line's last dispatches name their programs.
    pub(super) fn note_as(&mut self, what: impl FnOnce() -> String, named: bool) {
        if !cfg!(feature = "diagnostics") {
            return;
        }
        let pipeline = self.pipeline;
        let Some(r) = self.cmd.as_mut() else { return };
        let first = r.trail.notes.is_empty() && r.trail.earlier == 0;
        let same = std::mem::replace(&mut self.noted, pipeline) == pipeline;
        let note = if same && !first && !named {
            what()
        } else {
            let names = crate::pipeline::parts(pipeline)
                .map(|(s, _)| crate::pipeline::names(s))
                .unwrap_or_default();
            format!("{} [{names}]", what())
        };
        r.trail.push(note);
    }
}
