# The 7 minute talk

How to present `main_demo.html`. The slide order in that file is the order below,
so the deck can be run start to finish without skipping backwards.

Two sources shaped this: the hosts' brief on what is being judged, and the
capstone presentation skills deck (assertion and evidence, SCQA, one idea per
slide). Both are summarised at the bottom.

## Audience and aim

Judges are technical but not necessarily medical imaging people, and they see
many projects in a row. They are scoring idea, proposed solution, results and
presentation. What they should leave with, in one sentence:

> Image quality is not a safety metric, we measured how badly, and changing the
> objective fixes part of it.

Everything else on the deck exists to support that sentence or to show we know
where it stops being true.

## The narrative, as SCQA

| | |
|---|---|
| **Situation** | Cheap low field scanners are the realistic route to MRI access, and deep learning super resolution is the accepted way to make their images usable. |
| **Complication** | Super resolution is trained and judged on PSNR and SSIM, and those scores cannot see a small tumor. A forgery with the lesion deleted scores better than either model we train. |
| **Question** | Is a model trained this way quietly erasing lesions, and can a different objective stop it? |
| **Answer** | Yes to both, partly. Erasure falls 6.6 points at matched image quality on 70 unseen patients, the win is concentrated in small lesions exactly as the theory predicts, and it costs hallucination on a dial we can set. |

## The ten slides, with timings

Ten slides, no more. Timings assume 7 minutes total including the live demo, and
the cumulative column is where you should be as you leave the slide.

| # | Slide (assertion headline) | Evidence on screen | Time | Cum. |
|---|---|---|---|---|
| 1 | When sharper means blind | title, five names and roles | 0:10 | 0:10 |
| 2 | Cheap, low quality scanners are the only realistic way to widen MRI access | under 1 scanner per million vs 37, and 0.055 T | 0:40 | 0:50 |
| 3 | A forgery with the tumor painted out scores better than either reconstruction we train | 3 row table: forgery 28.8 dB / 0.978, baseline 24.5 / 0.778, ours 24.3 / 0.764 | 0:55 | 1:45 |
| 4 | We measure the erasure that a quality metric hides, then train against it | three contribution cards | 0:35 | 2:20 |
| 5 | Degrade a real scan, reconstruct it two ways, and ask one frozen detector what it can still find | four step pipeline | 0:45 | 3:05 |
| 6 | Super resolution recovers tumor the degradation destroys. Ours recovers more of it | the three bar erasure ladder, 62.2 / 58.0 / 51.3 | 1:15 | 4:20 |
| 7 | Large lesions are almost never lost. Small ones are the whole problem | erasure by lesion size | 0:50 | 5:10 |
| 8 | The lesions the baseline loses show up in 3D as empty blue shells | four 3D viewports, **this is the demo** | 0:50 | 6:00 |
| 9 | The next number we produce is the one that counts | four next steps | 0:35 | 6:35 |
| 10 | Five limits we would rather state than be asked | scope list, read two of them aloud | 0:25 | 7:00 |

## The backups, and how to reach them

Ten more slides sit off the running order: press **B** or click **Backup** in the
nav bar, then **B** or Escape to come back where you left off. The dots and the
counter only ever count the ten, so nothing here lengthens the talk.

| Backup | Use it when asked |
|---|---|
| The erasure and hallucination tradeoff is a dial | what does the fix cost, can you tune it |
| A rotating overlay | anything visual, or if a judge wants to see more of the demo |
| 2D evidence, three slices | show me a single case, not an average |
| The 3D volumes are restacked 2D predictions | is this a 3D model |
| The detector misses most small lesions even on an untouched scan | why is half still missed, is the segmenter the problem |
| Two flaws in our own pipeline cut our first headline in half | how do we know the numbers are sound |
| Who did what | who did what |
| Reproducing this | can we run it |

Know which backup answers which question before you go on. Fumbling for a slide
costs more credibility than not having one.

## What to say, per slide

Only the lines that are hard to improvise. Everything else is on the screen.

**1 and 2. The opening.** Roughly 50 seconds for both. The hook goes first,
before any introduction, because the first sentence is the only one the room is
guaranteed to hear.

> How would you know if the model that cleaned up your MRI had quietly removed
> your tumor?
>
> *(pause. let it sit)*
>
> We are five engineers, our names are on the slide, and we spent this hackathon
> answering that with numbers instead of opinions.
>
> *(slide 2)*
>
> Here is why the question is worth asking. Across much of sub Saharan Africa
> there is less than one MRI scanner per million people. In high income countries
> it is up to thirty seven. That gap does not get closed by buying more three
> tesla machines.
>
> *(pause)*
>
> Portable low field scanners do close it. They run at about fifty five
> millitesla, roughly a fiftieth of a hospital magnet, cheap enough to put in a
> district hospital. They are deployable, and their images are blurry and noisy.
>
> So the bottleneck moves from the scanner to the image it produces, and the
> accepted way to fix the image is deep learning super resolution: train on
> high field scans, then sharpen the cheap one. That bridge is where our question
> starts.

Optional lighter line if the room is stiff, in place of the second sentence:
"five engineers, one GPU and a suspicion." Use it or drop it, but do not stack it
on top of the hook. The hook works because nothing follows it for a beat.

Delivery: the hook is a question to the room, so look up and do not rush it. Give
the two scanner numbers slowly, one beat apart, and let the slide carry the
digits; you do not need to say "per million" twice. Do not read the three cards.

**3. The problem.** This is the slide the talk turns on, so land it slowly.
"We took the true scan, painted the tumor out with surrounding brain, and scored
that forgery against the original. It gets 28.8 dB. Our own best reconstruction
gets 24.3. On the metric the whole field trains on, deleting the tumor looks
like the better reconstruction. There is more measured error in ordinary blur
than in removing the tumor completely."

**4. Contribution.** Roughly 45 seconds. Lead with the objective, because it is
the idea; the other two cards are what make it checkable.

> So our fix is blunt. We heavily penalise the model for mistakes in the region
> where the tumor is. An error inside the lesion costs forty times what the same
> error costs anywhere else.
>
> *(pause)*
>
> It is a small change with a clever consequence. The model can no longer buy
> score by smoothing a lesion away, because the region it used to get away with
> ignoring is now the expensive one. We are pointing the loss at the medical
> abnormality instead of at the smoothness of the picture.
>
> Two things come with it. A second term supervises the detector's output rather
> than the pixels, so inventing a tumor is punished as well as losing one. And a
> readout: lesions erased, lesions fabricated, broken down by size, always
> measured against what the detector already misses on an untouched scan.
>
> No new architecture. Three small U-Nets. The contribution is what we optimise
> for, and an evaluation honest enough to show it. *[next speaker]* will show you
> how it runs.

If you are behind on time, cut to the first two paragraphs and the last line.
That still lands the idea and the hand off, and slide 5 covers the pipeline
anyway.

One wording caution: say the loss makes lesion error expensive, not that the
model "detects" the abnormality. The super resolution network is not a detector,
and a judge who knows that will pull the thread. If someone asks whether this is
just class weighting: yes, in mechanism, and the point is that nobody applies it
to super resolution because the quality metric hides what erasure costs. The
second term is what turns it from a fixed price into a dial, which is the first
backup slide.

**5. Method and architecture.** Roughly 45 seconds. Four steps, and step three
is the one that makes the comparison mean anything, so give it the emphasis.

> Four steps. We take a real high field scan and degrade it ourselves: crop
> k space by a factor of four, add Rician noise. Every blurry input therefore has
> an exact ground truth, and we never need a paired low field acquisition.
>
> Two networks then reconstruct it. Same U-Net, same data, same schedule. The
> only difference between them is the loss function.
>
> *(beat)*
>
> Both reconstructions go to one frozen tumor detector. The same detector every
> time. So when it finds less, that is caused by the image, not by a different
> ruler.
>
> Then we score twice: image quality inside the brain, and the two safety rates,
> lesions erased and lesions fabricated.
>
> One deliberate omission. No GAN. An adversarial loss rewards inventing
> plausible tissue, which is the exact failure we are trying to measure. We are
> not putting it inside our own instrument.

Numbers if asked: three small 2D U-Nets from one shared implementation, trained
from scratch, no pretrained weights, 128 by 128 slices, about fifty minutes per
configuration on one GPU, inference on CPU.

**6. The result.** Roughly 75 seconds. This is the slide the talk exists for.
Read the ladder top to bottom, then stop and deliver the matched quality line
slowly, because that is the finding.

> Seventy patients no model had seen, split by patient so not one slice of them
> was ever trained on. Read it top to bottom.
>
> In the degraded scan, the detector misses sixty two percent of enhancing lesion
> components. That is the cheap scan, before any reconstruction.
>
> Standard super resolution recovers some of them. Fifty eight.
>
> Ours recovers more. Fifty one point three. Six and a half points better than
> the baseline.
>
> *(pause)*
>
> Now the part to hold on to. The image quality of those two reconstructions is
> twenty four point five and twenty four point three decibels. Two tenths apart.
> By the metric the field trains on, those two images are equally good. One of
> them keeps considerably more tumor. That is the finding: image quality is not
> a safety metric.
>
> Two honest notes. It costs hallucination, false positives rise from zero point
> two seven to zero point three nine. And these are validation numbers. The
> ninety four test patients get one evaluation, once, and that is the number we
> will stand behind.

Do not skip the second honest note even if you are behind. Volunteering the
limit is what makes the rest of it credible, and it costs four seconds.

**7. The same result by lesion size.** Roughly 50 seconds. Bottom row first.
This slide exists to stop a judge misreading a rate near fifty percent.

> Same patients, now split by lesion size. Read the bottom row first.
>
> A lesion over two hundred pixels is missed about one percent of the time, by
> either model. Large tumors are not being erased. So nothing on the previous
> slide means half of all tumors vanish.
>
> The rate is driven by the top row. Seventy one percent of all components are
> under fifty pixels, and that is where the objective earns its keep: seventy
> eight point six, down to sixty nine point eight.
>
> *(beat)*
>
> And this is the mechanism confirming itself. The loss only helps where the
> structure is small enough for a pixel score to ignore it. We ran the identical
> pipeline on whole tumor, which is a tenth of the image, and it does nothing at
> all. Had it helped there too, we would have distrusted our own result.

**8. Demo.** Roughly 50 seconds, and most of it should be silence. One sentence
to teach the picture, then let them look.

> One held out patient, restacked from the per slice predictions.
>
> Blue is the true lesion. Where a model kept it, the lesion is filled in. Where
> a model lost it, you get a blue shell with nothing inside it.
>
> *(quiet. let them find one)*
>
> Ground truth, then ours, then the baseline, and the fourth panel is where the
> model is least sure of itself.
>
> One caveat we insist on: do not read volumes off this. The detector over reads
> even a clean scan, so this is here to be inspected, not counted.

If a judge wants it moving, the rotating overlay is the second backup slide.

**9. Next steps.** "The honest priority is not a better loss. It is one frozen
detector across all three objectives, then a single evaluation on the 94 test
patients, then abstaining where the model is unsure instead of returning a
confidently crisp image."

## Roles

The brief asks that each member's role be clear. Fill the `ROLES` dict at the
top of `main_demo.py`: the roles then appear next to the names on the title
slide, so they are covered without spending one of the ten. The backup roles
slide picks them up too, and unfilled entries show there in amber.

Suggested speaker split, so that all five present and nobody narrates a slide
they did not build:

| Slides | Section | Speaker |
|---|---|---|
| 1 to 3 | the problem | |
| 4 to 5 | contribution and method | |
| 6 to 7 | results | |
| 8 | the 3D demo | |
| 9 to 10 | next steps and scope | |

Hand offs cost time. Name the next speaker at the end of your last sentence
rather than pausing between slides.

## Questions to have answers ready for

The presentation skills deck is blunt about this: not knowing a hard question is
acceptable, not knowing a simple one is not. These are the simple ones.

**Is 6.6 points clinically meaningful?** We do not claim it is. It is a
measurable effect in the right direction at matched image quality, on 70 unseen
patients, patient level paired bootstrap p of about 0.01. Whether it changes a
diagnosis is a clinical question we are not equipped to answer, and the
per patient spread matters: on the earlier held out run, 11 of 71 patients got
worse.

**Half the lesions are still missed. Why is that acceptable?** It is not, and
the backup slide has the reason. Run the detector on the untouched original
scan, no degradation and no reconstruction at all, and it already misses 45.5%
of components and 65% of the small ones. Most of the absolute rate is the
detector's floor, not damage from super resolution. The part we cause is the
excess, and our objective removes about 71% of it.

**Then is the segmenter the real problem?** Yes, and we say so on the deck. A
stronger detector would buy more than a better loss. Our claim is about the
comparison between two objectives measured with the same instrument, which is
valid even where the instrument is weak, and a weak instrument compresses the
gap rather than inventing it.

**Is the degradation realistic?** Partly. K space truncation plus Rician noise
reproduces resolution loss and noise. It does not reproduce the contrast change
of a real 0.055 T scanner. That is why the next step is a real low field
acquisition or BraTS Africa, and why we call this a proof of concept.

**What does the fix cost?** Hallucination: false positive rate rises from 0.266
to 0.387. The second finding is that this is adjustable rather than fixed, and
the dial is the first backup slide. A
segmentation consistency term takes the added hallucination from +0.121 to
+0.017 while giving up about half the erasure win, so a screening setting that
cannot afford false alarms has a setting available.

**Why should we believe the numbers?** Because we broke them ourselves first.
An audit found a dropout leak that meant every published safety number had been
scored on stochastic reconstructions, and lesions counted per slice rather than
per patient, which overstated our precision by 1.7 times. Fixing both cut the
headline gap from 5.5 points to 2.4. Every run is in `EXPERIMENTS.md` with its
data source next to it, and that file is append only.

**Why not a GAN, or a diffusion model?** They optimise perceptual realism, which
rewards inventing plausible tissue. That is the failure mode we are measuring,
so putting it inside the instrument would make the study unable to detect it.
The finding here is about the objective, not the architecture, and it applies to
any model trained on a distortion metric.

**Why not a 3D model?** Data and compute. Three small 2D U Nets train on one
GPU in under an hour and run inference on CPU, which matters for the deployment
setting we are motivated by. The 3D volumes on the deck are restacked 2D
predictions and we label them as such.

**What did it cost?** One free tier GPU box, about 50 minutes per training
configuration, and no pretrained weights. Public data, MSD Task01, no
registration wall. Inference runs on CPU.

**What is left, and when?** The shared detector rerun and the single test
evaluation are written and queued, blocked only on a GPU box that survives an
hour. Uncertainty gated abstention needs no GPU at all and is the most
defensible improvement still available.

## The two inputs

**From the hosts.** 7 minutes including any demo, plus 5 minutes of questions.
The presentation should expose the idea, the proposed solution, clear technical
choices and the results with quantitative metrics. Share screen for the live
demo and have everything ready in advance. All team members present, each
member's role clear. Judged on: idea (originality, social impact, focus),
proposed solution (technical accuracy, complexity, contribution), results
(proficiency, performance, proper evaluation), presentation (clarity, quality,
timing, group work).

**From the capstone presentation skills deck.** The presentation is a shop
window, not the report: include only what makes the audience curious. Structure
with SCQA. Use assertion and evidence rather than topic headlines and bullets,
because bullets are not remembered: every headline is a complete sentence
stating the message, and one visual under it carries the proof. One idea per
slide. Label a graph with its conclusion, not its contents. Short parallel
phrases, at most four or five, never full sentences. Only "appear" animation.
Slow down, do not read the slides, practise against a timer. Be ready for
questions on cost, resources, timeline, expected benefits and why you rejected
the alternatives.

Where this deck knowingly departs: the backup slides are dense, because they
are read rather than presented. The ten talk slides carry a headline, one piece
of evidence and at most two grey footnote lines. If you find yourself wanting to
add a paragraph to one of the ten, it belongs in a backup slide or in this
file.
