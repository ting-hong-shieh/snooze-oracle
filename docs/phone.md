# Put the plan on your phone

The planner writes `out/plan.json` on your computer. These steps serve that file on your
home network and let a phone shortcut turn it into alarms: a first ring 30 minutes before
the wake time and a final alarm at the wake time.

The times come from a replay of an August 2026 morning (see the README). They are clock
times, so the alarms ring at that time on your phone wherever you are.

Nothing here was tested on a phone by me; check each step on yours.

## 1. Make a plan

```bash
cd ~/projects/competition/hackathon/snooze-oracle
.venv/bin/python -m src.agent plan --scenario commute --from "Cobble Hill" --to "Midtown North" --deadline 09:00
```

## 2. Serve `out/` on the local network

This serves only the `out/` folder (`plan.json` and `alarms.ics`), nothing else in the repo.

In WSL:

```bash
cd ~/projects/competition/hackathon/snooze-oracle && python3 -m http.server 8765 --bind 0.0.0.0 --directory out
```

WSL here uses NAT networking, so the phone cannot reach WSL directly. Forward the port
from Windows once, in PowerShell **as administrator**:

```powershell
$wsl = (wsl hostname -I).Trim().Split(" ")[0]
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8765 connectaddress=$wsl connectport=8765
New-NetFirewallRule -DisplayName "Snooze Oracle 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private -RemoteAddress LocalSubnet
```

- The firewall rule only opens the port on a **Private** network to devices on the same
  subnet. Your Wi-Fi must be set to Private in Windows settings.
- Find your computer's address with `ipconfig` (the IPv4 address of the Wi-Fi adapter,
  usually `192.168.x.x`). Below it is `<your-PC-IP>`.
- WSL's address changes after a restart. If the phone stops connecting, delete the
  forward (below) and add it again.

Check it: on the phone, open `http://<your-PC-IP>:8765/plan.json` in Safari or Chrome. You
should see the JSON.

When you are done, remove both:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8765
Remove-NetFirewallRule -DisplayName "Snooze Oracle 8765"
```

## 3. iPhone: a Shortcut that sets the alarms

Create a new shortcut in the Shortcuts app and add these actions in order.

1. **Text**: `http://<your-PC-IP>:8765/plan.json`
2. **Get Contents of URL** (input: the Text).
3. **Get Dictionary Value**: get *Value* for *first_ring* in *Contents of URL*.
4. **Get Dates from Input** (input: that Dictionary Value). Rename the result
   "First ring" (tap the variable → Rename).
5. **Get Dictionary Value**: *wake_time* in *Contents of URL*.
6. **Get Dates from Input** (input: that value). Rename it "Wake".
7. **Create Alarm** at *First ring*, label `Snooze Oracle: first ring`.
8. **Create Alarm** at *Wake*, label `Snooze Oracle: get up`.

Run it once. Open the Clock app and check that both alarms exist and are on.

### Snooze on the final alarm

The final alarm is the one you should not snooze. I found no documentation that the
Create Alarm action can turn snooze off: Apple's guide mentions snooze only as an
automation trigger ("Is Snoozed"), and the action reference I found lists only the time,
the label and repeat. Check on your phone: in the Create Alarm action, tap the arrow or
"Show More". If you see a snooze switch, turn it off in step 8 and skip the fallback.

Fallback: after step 8, add extra alarms one minute apart for five minutes.

9. **Repeat** 5 times:
   - **Adjust Date**: add *Repeat Index* minutes to *Wake*.
   - **Create Alarm** at *Adjusted Date*, label `Snooze Oracle: get up`.
10. **End Repeat**.

The first ring can keep the normal snooze. Pressing snooze there gives the Clock app's
snooze length (9 minutes by default), which the plan's 30-minute window allows three
times.

Shortcuts always creates new alarms, so delete the old Snooze Oracle alarms in the Clock
app before running it for another morning.

### Without a shortcut

AirDrop or open `http://<your-PC-IP>:8765/alarms.ics` and add it to Calendar. You get two
calendar alerts at the same times. They are notifications, not alarms, so they do not ring
through silent mode.

## 4. Android

The standard Clock app can be set from other apps through Android's "set alarm" request.
Two ways:

- **MacroDroid or Tasker**: an *HTTP request* to `http://<your-PC-IP>:8765/plan.json`,
  read `first_ring` and `wake_time` from the JSON, then a *Set Alarm* action for each.
  For the final alarm, either turn snooze off in the Clock app afterwards or add alarms at
  wake_time +1 to +5 minutes.
- **Calendar**: open `alarms.ics` and add it to Google Calendar for two notifications (not
  alarms).

As with the iPhone steps, I could not test these on a device.
